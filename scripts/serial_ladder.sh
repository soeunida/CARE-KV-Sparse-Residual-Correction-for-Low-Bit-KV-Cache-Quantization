#!/usr/bin/env bash
# Strictly serial ladder re-run (one model at a time) on a single GPU to avoid
# the parallel-contention failures. GPU via SERIAL_GPU env (default 1).
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
export CUDA_VISIBLE_DEVICES=${SERIAL_GPU:-1}
RD=results/blockgtq_carekv
OUT=$RD/adapter_ladder_n32.csv

MODELS="openlm-research/open_llama_3b_v2 deepseek-ai/deepseek-llm-7b-base openlm-research/open_llama_7b_v2 upstage/SOLAR-10.7B-v1.0 NousResearch/Llama-2-13b-hf"

for model in $MODELS; do
  tag=$(basename "$model")
  echo "[serial] === $tag START $(date +%H:%M:%S) ==="
  python run_bgtq_adapter.py --model-id "$model" --seq-len 512 \
    --num-samples 32 --append-csv "$OUT" > "$RD/serial_${tag}.log" 2>&1
  rc=$?
  tail=$(grep -E "DONE|Traceback|Error" "$RD/serial_${tag}.log" | grep -v "^  BlockGTQ:" | tail -1)
  echo "[serial] === $tag END rc=$rc $(date +%H:%M:%S) :: $tail ==="
done
echo "[serial] ALL DONE $(date +%H:%M:%S)"
