#!/usr/bin/env bash
#
# scripts/run_wikitext2_ppl.sh
# ----------------------------
# Phase I — WikiText-2 PPL across modes at the paper-best CARE-KV config.
#
# Env knobs:
#   SEQ_LEN        (default 128)
#   NUM_SAMPLES    (default 4)
#   OUT_DIR        (default results/paper_eval_<ts>/ppl_dataset)
#   INCLUDE_INT2   (default 0)  — set 1 to also run INT2 carekv_stored
#
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

PAPER_DIR="${PAPER_DIR:-results/paper_eval_20260529_015053}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
INCLUDE_INT2="${INCLUDE_INT2:-0}"
OUT_DIR="${OUT_DIR:-$PAPER_DIR/ppl_dataset}"
CSV="$OUT_DIR/wikitext2_ppl.csv"
LOGS="$OUT_DIR/logs"

mkdir -p "$OUT_DIR" "$LOGS"
rm -f "$CSV"

export DATASET_NAME=wikitext
export DATASET_CONFIG=wikitext-2-raw-v1
export DATASET_SPLIT=test
export MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Paper-best CARE-KV knobs (locked) — see CLAUDE.md §2 (promoted 2026-07-15).
export CAREKV_PACKED_BASE=1
export CAREKV_SCALE_QUANT=int8
export CAREKV_PREFILL_RESIDUAL_KIND=both
export CAREKV_ROUTE_POLICY=joint
export CAREKV_SCORE_NORMALIZE=1
# combined_kvscore selector is vectorized-only (cached router ignores KSCORE); §10.
export CAREKV_CORRECTION_IMPL=vectorized
export CAREKV_K_CORRECTION_MODE=exact   # exact softmax renorm, not 1st-order Jacobian
export CAREKV_KSCORE_LIVE=1             # combined_kvscore K+V selector
export CAREKV_BUDGET_POLICY=uniform
export STORE_ABS_K=2
export STORE_ABS_V=4
export READ_ABS_K=2
export READ_ABS_V=2
export CAREKV_DEBUG_STATS=1

# To reproduce the pre-2026-07-15 paper-best (linear / cached / current selector):
#   export CAREKV_CORRECTION_IMPL=cached CAREKV_K_CORRECTION_MODE=linear
#   unset CAREKV_KSCORE_LIVE

run_one() {
  local mode="$1"; local bits="$2"; local label="$3"
  echo "=== $label ==="
  MODE="$label" BASE_BITS="$bits" SEQ_LEN="$SEQ_LEN" NUM_SAMPLES="$NUM_SAMPLES" \
    python eval_ppl_dataset.py --mode "$mode" --base-bits "$bits" \
    --mode-label "$label" --append-csv "$CSV" \
    2>&1 | tee "$LOGS/$label.log"
}

# fp16 baseline
run_one fp16          16 "fp16"

# base_quant ladder
run_one base_quant     4 "base_quant_int4"
run_one base_quant     3 "base_quant_int3"
run_one base_quant     2 "base_quant_int2"

# carekv_stored at the paper-best config
run_one carekv_stored  3 "carekv_stored_int3_optimized"
if [ "$INCLUDE_INT2" = "1" ]; then
  run_one carekv_stored 2 "carekv_stored_int2_optimized"
fi

echo
echo "==================================================="
echo "  CSV : $CSV"
echo "==================================================="
