#!/usr/bin/env bash
set -euo pipefail

cd /home/soeun/CARE_KV/care_kv
mkdir -p results

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODEL_ID=${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
export PYTHONPATH=${PYTHONPATH:-/home/soeun}

echo "===== FP16 ====="
MODE=fp16 \
USE_CACHE=0 \
DO_SAMPLE=0 \
MAX_NEW_TOKENS=64 \
python run_llama_carekv.py | tee results/gen_fp16_sanity.txt

for B in 4 3 2; do
  echo "===== INT$B base_quant ====="
  MODE=carekv \
  CAREKV_RETURN=care \
  CAREKV_PREFILL_MODE=base_quant \
  BASE_BITS=$B \
  USE_CACHE=0 \
  DO_SAMPLE=0 \
  MAX_NEW_TOKENS=64 \
  python run_llama_carekv.py | tee results/gen_basequant_int${B}_sanity.txt
done

for B in 3 2; do
  echo "===== INT$B CAREKV V-only ====="
  MODE=carekv \
  CAREKV_RETURN=care \
  CAREKV_PREFILL_MODE=carekv \
  CAREKV_PREFILL_RESIDUAL_KIND=v \
  BASE_BITS=$B \
  CAREKV_PREFILL_RESIDUAL_RATIO=0.05 \
  CAREKV_K_CORRECTION_SCALE=0.0 \
  USE_CACHE=0 \
  DO_SAMPLE=0 \
  MAX_NEW_TOKENS=64 \
  python run_llama_carekv.py | tee results/gen_carekv_int${B}_Vonly_sanity.txt
done
