#!/usr/bin/env bash
# Complete the improved ladder under a SINGLE config (fine_rb8), into a CSV that
# records the config per row (improved_ladder_v2.csv).
#
# Why not append to improved_ladder.csv: its Llama-2-13b row came from an
# unidentified rerun after the fine_rb8 C-arm OOMed, so that file already mixes
# configs and has no column to tell them apart. 13B is therefore re-run here too.
#
# Resumable: models already present in the CSV are skipped, so a reboot (two
# happened 07-20) only costs the in-flight model, not the queue.
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
# fine_rb8's rk8/rv8 read budgets are what OOMed 13B before; reduce fragmentation
# and shard the big models across 2 GPUs (see run1 callers).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8
export CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8
export CAREKV_CFG_NAME=fine_rb8

RD=results/blockgtq_carekv
OUT=$RD/improved_ladder_v2.csv

run1 () {  # visible_devices model logtag [devmap]
  local dev=$1 model=$2 lg=$3 dm=${4:-}
  if [ -f "$OUT" ] && grep -q "^${model}," "$OUT"; then
    echo "[fine] === $lg SKIP (already in $OUT) $(date +%m-%d_%H:%M)"; return 0
  fi
  echo "[fine] >>> $lg START dev=$dev $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$dev CAREKV_DEVICE_MAP=$dm python run_bgtq_adapter.py \
    --model-id "$model" --seq-len 512 --num-samples 32 --append-csv "$OUT" \
    > "$RD/fine_${lg}.log" 2>&1
  echo "[fine] >>> $lg END rc=$? $(date +%m-%d_%H:%M) :: $(grep 'DONE' "$RD/fine_${lg}.log" | grep -v '^  BlockGTQ:' | tail -1)"
}

# Three single-model queues in parallel; 2 GPUs each (device_map=auto) because
# fine_rb8 C-arm OOMed on one 48GB card.
( run1 0,1 NousResearch/Llama-2-13b-hf         13b        auto ) &
( run1 2,3 deepseek-ai/deepseek-llm-7b-base    deepseek   auto ) &
( run1 4,5 openlm-research/open_llama_7b_v2    openllama  auto ) &
wait
echo "[fine] 7B/13B queues done — starting Yi-34B (4-GPU) $(date +%m-%d_%H:%M)"
run1 0,1,2,3 01-ai/Yi-34B Yi-34B auto
echo "[fine] ALL DONE $(date +%m-%d_%H:%M)"
