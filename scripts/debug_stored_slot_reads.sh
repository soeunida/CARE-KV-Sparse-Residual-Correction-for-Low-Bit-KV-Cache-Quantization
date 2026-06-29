#!/usr/bin/env bash
#
# scripts/debug_stored_slot_reads.sh
# -----------------------------------
# Targeted reproducer for the carekv_stored zero-reads issue.
# Tests A-E from the user spec and prints, per case:
#   PPL, store_budget, read_budget,
#   stored V/K slot counts, read V/K slot counts,
#   routing candidate counts, mean topk,
#   whether correction was applied,
#   mean |ΔO_V|, mean |ΔO_K|.
#
# Writes results to: results/debug_stored_slot_reads_<ts>.txt
#
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

TS=$(date +%Y%m%d_%H%M%S)
OUT="results/debug_stored_slot_reads_${TS}.txt"
mkdir -p results
SEQ_LEN="${SEQ_LEN:-128}"
MODEL_ID="${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"

python tools/debug_stored_slot_reads.py --seq-len "$SEQ_LEN" --out "$OUT"
echo
echo "=== output → $OUT ==="
cat "$OUT"
