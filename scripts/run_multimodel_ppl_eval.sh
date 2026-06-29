#!/usr/bin/env bash
#
# scripts/run_multimodel_ppl_eval.sh
# -----------------------------------
# Phase I-3 — run WikiText-2 PPL across multiple LLaMA-style models.
# Skips gated/missing models cleanly.
#
# Env:
#   MODELS         — space-separated MODEL_IDs (default: TinyLlama + JackFram-160m)
#   SEQ_LEN        — sequence length (default 128 to stay tractable under
#                    carekv_stored's Python prefill loop)
#   NUM_SAMPLES    — windows per model (default 4)
#   OUT_DIR
#
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

PAPER_DIR="${PAPER_DIR:-results/paper_eval_20260529_015053}"
SEQ_LEN="${SEQ_LEN:-128}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MODELS="${MODELS:-TinyLlama/TinyLlama-1.1B-Chat-v1.0 JackFram/llama-160m}"
OUT_DIR="${OUT_DIR:-$PAPER_DIR/ppl_dataset}"
CSV="$OUT_DIR/multimodel_wikitext2.csv"
LOGS="$OUT_DIR/multimodel_logs"

mkdir -p "$OUT_DIR" "$LOGS"
rm -f "$CSV"

export DATASET_NAME=wikitext
export DATASET_CONFIG=wikitext-2-raw-v1
export DATASET_SPLIT=test

# Paper-best CARE-KV knobs (locked)
export CAREKV_PACKED_BASE=1
export CAREKV_SCALE_QUANT=int8
export CAREKV_PREFILL_RESIDUAL_KIND=both
export CAREKV_ROUTE_POLICY=joint
export CAREKV_SCORE_NORMALIZE=1
export CAREKV_CORRECTION_IMPL=cached
export CAREKV_BUDGET_POLICY=uniform
export STORE_ABS_K=2; export STORE_ABS_V=4
export READ_ABS_K=2;  export READ_ABS_V=2
export CAREKV_DEBUG_STATS=1

for MODEL_ID in $MODELS; do
  echo "============================================================"
  echo "  MODEL_ID=$MODEL_ID"
  echo "============================================================"
  MODEL_TAG=$(echo "$MODEL_ID" | tr '/' '_')

  for cell_mode in fp16 base_quant carekv_stored; do
    LABEL="$cell_mode"; BITS=3
    [ "$cell_mode" = "fp16" ] && LABEL="fp16"
    [ "$cell_mode" = "base_quant" ] && LABEL="base_quant_int3"
    [ "$cell_mode" = "carekv_stored" ] && LABEL="carekv_stored_int3_optimized"

    echo "--- $MODEL_TAG / $LABEL ---"
    MODEL_ID="$MODEL_ID" MODE="$LABEL" BASE_BITS=$BITS \
      SEQ_LEN="$SEQ_LEN" NUM_SAMPLES="$NUM_SAMPLES" \
      python eval_ppl_dataset.py \
        --model-id "$MODEL_ID" --mode "$cell_mode" --base-bits $BITS \
        --mode-label "$LABEL" --append-csv "$CSV" \
        2>&1 | tee "$LOGS/${MODEL_TAG}_${LABEL}.log" || \
      echo "[multimodel] $MODEL_TAG / $LABEL FAILED" >> "$CSV.failures"
  done
done

echo
echo "============================================================"
echo "  CSV   : $CSV"
echo "  LOGS  : $LOGS/"
echo "============================================================"
