#!/usr/bin/env bash
#
# scripts/bench_latency.sh
# -------------------------
# Phase H — decode latency benchmark.
#
# Env:
#   MODEL_ID            (default TinyLlama)
#   PROMPT_LENS         (default "128 512")
#   NEW_TOKENS          (default 64; reduced via CAREKV_NEW_TOKENS for carekv_stored)
#   CAREKV_NEW_TOKENS   (default 8)
#   OUT_DIR             (default results/latency_<ts>)
#
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

TS=$(date +%Y%m%d_%H%M%S)
MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PROMPT_LENS="${PROMPT_LENS:-128 512}"
NEW_TOKENS="${NEW_TOKENS:-64}"
CAREKV_NEW_TOKENS="${CAREKV_NEW_TOKENS:-8}"
OUT_DIR="${OUT_DIR:-results/latency_${TS}}"

mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/latency.csv"
LOG="$OUT_DIR/bench.log"

echo "=== bench_latency  model=$MODEL_ID  prompts=$PROMPT_LENS  new_tokens=$NEW_TOKENS  carekv_new=$CAREKV_NEW_TOKENS ===" \
  | tee "$LOG"

python tools/bench_latency.py \
  --model-id "$MODEL_ID" \
  --prompt-lens $PROMPT_LENS \
  --new-tokens "$NEW_TOKENS" \
  --carekv-new-tokens "$CAREKV_NEW_TOKENS" \
  --out-csv "$CSV" 2>&1 | tee -a "$LOG"

echo
echo "============================================================"
echo "  CSV : $CSV"
echo "  LOG : $LOG"
echo "============================================================"
