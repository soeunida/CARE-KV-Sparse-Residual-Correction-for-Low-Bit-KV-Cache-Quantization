#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_long_context_ppl.sh — SL >= 4096 long-context evaluation for CARE-KV.
#
# Motivation: the existing paper PPL runs are SL <= 128 (and 7B validation at
# SL <= 1024), where the KV cache is NOT the bottleneck. This script targets the
# regime where the KV cache *is* the bottleneck (SL >= 4096) and reports the
# quantities that are feasible there:
#
#   * fp16 PPL (real forward, model-agnostic)                        [real]
#   * BaseQuant INT4 / INT3 KV PPL via a model-agnostic per-group    [real]
#     fake-quant hook (KIVI-style: per-channel K, per-token V) —
#     NOT the slow CARE-KV Python-loop prefill, so it runs at 4096.
#   * fp16 / BaseQuant / CARE-KV KV-memory vs SL (analytical estimator)
#                                                                     [analytical]
#     — this is the direct "KV cache is the bottleneck" evidence.
#
# NOT run here (documented prototype blocker — see the generated summary):
#   * CARE-KV native carekv_stored / per-group base_quant PREFILL at SL>=4096:
#     the per-(layer,kv_head,token) Python-loop prefill is multi-hour and the
#     DynamicCache dummy-fp16 inflation OOMs a 49 GB GPU at SL=4096. Measured
#     at SL=1024 (DeepSeek-7B, N=4): base_quant 1143 s / 42 GB, carekv_stored
#     2921 s / 34 GB. CARE-KV PPL at long SL is projected via its short-context
#     anchor + analytical memory (unblocks once joint+both is vectorized).
#
# Model: DeepSeek-7B (native context 4096). TinyLlama/llama-160m cap at 2048.
# ---------------------------------------------------------------------------
set -euo pipefail

cd /home/soeun/CARE_KV/care_kv
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

MODEL_ID="${MODEL_ID:-deepseek-ai/deepseek-llm-7b-base}"
SEQ_LENS="${SEQ_LENS:-2048 4096}"          # >= native-context edge for DeepSeek-7B (4096)
NUM_SAMPLES="${NUM_SAMPLES:-8}"
GPU="${GPU:-1}"
OUT_DIR="${OUT_DIR:-results/long_context_ppl}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
mkdir -p "$OUT_DIR"

echo "[long-ctx] model=$MODEL_ID seq_lens=[$SEQ_LENS] N=$NUM_SAMPLES gpu=$CUDA_VISIBLE_DEVICES"

# --- 1. fp16 + INT4/INT3 KV PPL + analytical KV-memory table, per SL ---
for SL in $SEQ_LENS; do
  echo "[long-ctx] === SL=$SL : fp16 + INT4/INT3 KV PPL + memory table ==="
  python tools/eval_long_context_kv_memory.py \
    --model "$MODEL_ID" --seq-len "$SL" --num-samples "$NUM_SAMPLES" \
    --out-csv "$OUT_DIR/kv_memory_sl${SL}.csv"
done

# --- 2. Canonical-driver fp16 cross-check at the max SL (validates eval_ppl_dataset
#        handles SL>=4096; carekv/base_quant left to the blocked path). ---
MAXSL=$(echo "$SEQ_LENS" | tr ' ' '\n' | sort -n | tail -1)
echo "[long-ctx] === canonical fp16 cross-check at SL=$MAXSL ==="
MODEL_ID="$MODEL_ID" python eval_ppl_dataset.py \
  --model-id "$MODEL_ID" --mode fp16 --seq-len "$MAXSL" --num-samples "$NUM_SAMPLES" \
  --append-csv "$OUT_DIR/long_context_canonical_fp16.csv"

# --- 3. Regenerate the honest markdown summary from the CSVs. ---
python tools/summarize_long_context.py --out-dir "$OUT_DIR" || \
  echo "[long-ctx] (summary generator not found; CSVs written to $OUT_DIR)"

echo "[long-ctx] done -> $OUT_DIR"
