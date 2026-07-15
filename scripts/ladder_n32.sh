#!/usr/bin/env bash
# Model-size ladder for Block-GTQ ⊕ CARE-KV, SL512 N=32.
# Waits for the in-flight Mistral SL1024 N32 cell to finish (frees GPU1), then
# runs a size ladder across GPUs 1,4,5 (each GPU processes its queue serially).
# GPU6 is left alone (user's exact_kcorr job); GPUs 0,2,3 are another user's.
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
RD=results/blockgtq_carekv
OUT=$RD/adapter_ladder_n32.csv

echo "[ladder] waiting for Mistral SL1024 N32 to finish… $(date +%H:%M:%S)"
until grep -q "DONE" $RD/n32_mistral1024.log 2>/dev/null; do sleep 60; done
echo "[ladder] in-flight run done — starting ladder $(date +%H:%M:%S)"

run_queue () {  # gpu  model1 model2 ...
  local gpu=$1; shift
  for model in "$@"; do
    local tag=$(basename "$model")
    echo "[ladder] >>> gpu$gpu $tag START $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu python run_bgtq_adapter.py --model-id "$model" \
      --seq-len 512 --num-samples 32 --append-csv "$OUT" \
      > "$RD/ladder_${tag}.log" 2>&1
    echo "[ladder] >>> gpu$gpu $tag END   $(date +%H:%M:%S)"
  done
}

# Three serial queues, one per free GPU, run in parallel.
run_queue 5 JackFram/llama-160m openlm-research/open_llama_3b_v2 NousResearch/Llama-2-13b-hf &
run_queue 4 openlm-research/open_llama_7b_v2 upstage/SOLAR-10.7B-v1.0 &
run_queue 1 01-ai/Yi-6B deepseek-ai/deepseek-llm-7b-base &
wait
echo "[ladder] ALL DONE $(date +%H:%M:%S)"
echo "=== ladder CSV ==="; cat "$OUT"
