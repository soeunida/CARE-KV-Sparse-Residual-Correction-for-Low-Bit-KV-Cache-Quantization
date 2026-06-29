#!/usr/bin/env bash
#
# scripts/run_all_paper_eval.sh
# ------------------------------
# Phase L unified runner. Re-executes the curated paper eval matrix into
# an existing paper_eval_<timestamp> directory.
#
# Compute envelope (TinyLlama-1.1B, single A100/H100-class GPU):
#   - I-1 WikiText-2 N=16 SL=128:                      ~1.8 h (carekv_stored cell dominates)
#   - J   Long-context ctx=128 n_pairs=6 trials=5      ~1.7 h (3 tasks × 3 modes)
#   - K   Per-layer diagnostics + 8 summary figures:   ~1 min
#   - All other steps:                                 seconds–minutes
#
# Defaults reproduce the paper-headline run. Drop to smoke values
# (WT2_NUM_SAMPLES=4, LONG_CTX_TRIALS=2) for a fast sanity sweep.
#
# Env knobs:
#   PAPER_DIR              — paper dir to fill (default: results/paper_eval_20260529_015053)
#   WT2_NUM_SAMPLES        — WikiText-2 windows (default 16 = paper N)
#   WT2_SEQ_LEN            — WT-2 window length (default 128)
#   LONG_CTX_TRIALS        — long-context trials per cell (default 5)
#   LONG_CTX               — long-context target length (default 128 — fp16-solvable on TinyLlama)
#   LONG_CTX_NUM_PAIRS     — kv-retrieval pair count (default 6)
#   LONG_CTX_MODES         — long-ctx modes (default fp16,base_quant_int3,carekv_int3_both)
#   LONG_CTX_TASKS         — long-ctx tasks (default kv_retrieval,boundary,copy)
#   SKIP_HEAVY             — set 1 to skip carekv cells everywhere
#   TRY_USE_CACHE          — set 1 to also include use_cache=True generation
#
set -euo pipefail
cd "$(dirname "$0")/.."
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

PAPER_DIR="${PAPER_DIR:-results/paper_eval_20260529_015053}"
WT2_NUM_SAMPLES="${WT2_NUM_SAMPLES:-16}"
WT2_SEQ_LEN="${WT2_SEQ_LEN:-128}"
LONG_CTX_TRIALS="${LONG_CTX_TRIALS:-5}"
LONG_CTX="${LONG_CTX:-128}"
LONG_CTX_NUM_PAIRS="${LONG_CTX_NUM_PAIRS:-6}"
LONG_CTX_MODES="${LONG_CTX_MODES:-fp16,base_quant_int3,carekv_int3_both}"
LONG_CTX_TASKS="${LONG_CTX_TASKS:-kv_retrieval,boundary,copy}"
SKIP_HEAVY="${SKIP_HEAVY:-0}"
TRY_USE_CACHE="${TRY_USE_CACHE:-0}"

[ -d "$PAPER_DIR" ] || { echo "ERROR: PAPER_DIR=$PAPER_DIR does not exist" >&2; exit 1; }
mkdir -p "$PAPER_DIR"/{logs,ppl,sweeps,ablations,memory,latency,figures,generation,summaries,ppl_dataset,long_context}

LOGS="$PAPER_DIR/logs"

# ── A. compile + tests ──────────────────────────────────────────
echo "=== A. compile + tests ==="
for f in __init__.py cache.py quantizer.py residual_store.py residual_router.py attention.py layer.py llama_patch.py utils.py tests/test_carekv_v2.py; do
  python -m py_compile "$f" && echo "OK $f"
done | tee "$LOGS/A_compile.log"

python -m pytest -q tests/test_carekv_v2.py 2>&1 | tee "$LOGS/A_tests.log" || {
  echo "TESTS FAILED — stopping" >&2; exit 1;
}

# ── I-1. WikiText-2 paper eval ──────────────────────────────────
echo "=== I-1. WikiText-2 PPL (NUM_SAMPLES=$WT2_NUM_SAMPLES, SEQ_LEN=$WT2_SEQ_LEN) ==="
PAPER_DIR="$PAPER_DIR" SEQ_LEN="$WT2_SEQ_LEN" NUM_SAMPLES="$WT2_NUM_SAMPLES" \
  INCLUDE_INT2=0 OUT_DIR="$PAPER_DIR/ppl_dataset" \
  bash scripts/run_wikitext2_ppl.sh 2>&1 | tee "$LOGS/I1_wikitext2.log"

# ── J. Long-context retrieval (TinyLlama-tractable config) ──────
if [ "$SKIP_HEAVY" = "0" ]; then
  echo "=== J. Long-context (trials=$LONG_CTX_TRIALS, ctx=$LONG_CTX, n_pairs=$LONG_CTX_NUM_PAIRS) ==="
  python scripts/run_long_context_retrieval.py \
    --out-csv "$PAPER_DIR/long_context/long_context_retrieval.csv" \
    --num-trials "$LONG_CTX_TRIALS" --ctx-target "$LONG_CTX" \
    --num-pairs "$LONG_CTX_NUM_PAIRS" \
    --tasks "$LONG_CTX_TASKS" --modes "$LONG_CTX_MODES" \
    2>&1 | tee "$LOGS/J_long_context.log" || echo "(long-context cell warned)"
fi

# ── K-a. Per-layer diagnostic figures ───────────────────────────
echo "=== K-a. Per-layer diagnostic figures ==="
python tools/paper_eval.py figures --out-dir "$PAPER_DIR/figures" \
  2>&1 | tee "$LOGS/Ka_layer_figures.log" || echo "(per-layer figures warned)"

# ── K-b. Paper summary figures ──────────────────────────────────
echo "=== K-b. Paper summary figures ==="
python tools/make_paper_figures.py --paper-dir "$PAPER_DIR" \
  2>&1 | tee "$LOGS/Kb_paper_figures.log"

# ── K-c. 3D activation distribution figures (Channel × Token × |value|) ──
echo "=== K-c. 3D activation distribution figures ==="
python tools/make_3d_activation_figures.py \
  --model-id "${KC_MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
  --out-dir "$PAPER_DIR/figures" \
  --layers ${KC_LAYERS:-0 11 21} \
  --seq-len "${KC_SEQ_LEN:-512}" \
  --max-tokens "${KC_MAX_TOKENS:-256}" \
  --max-channels "${KC_MAX_CHANNELS:-512}" \
  2>&1 | tee "$LOGS/Kc_activation_3d.log"

# ── K-d. CARE-KV before/after 3D distribution figures (|value| surfaces) ──
# fp16 → INT3 base_quant → INT3 CARE-KV (base_quant + stored residuals)
echo "=== K-d. Before/after 3D distribution figures ==="
python tools/make_before_after_3d_figures.py \
  --model-id "${KD_MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
  --out-dir "$PAPER_DIR/figures" \
  --layers ${KD_LAYERS:-0 11 21} \
  --seq-len "${KD_SEQ_LEN:-512}" \
  --max-tokens "${KD_MAX_TOKENS:-256}" \
  --max-channels "${KD_MAX_CHANNELS:-512}" \
  --base-bits "${KD_BASE_BITS:-3}" \
  --store-abs-k "${KD_STORE_ABS_K:-2}" \
  --store-abs-v "${KD_STORE_ABS_V:-4}" \
  2>&1 | tee "$LOGS/Kd_before_after_3d.log"

# ── K-e. CARE-KV error-decomposition figures (focuses the CARE-KV effect) ──
# 2D heatmaps are the primary paper figure; 3D scatter is a sparse,
# readable secondary view. Default mode is "all" so a single invocation
# produces error-decomposition + visible-error + clean-error outputs.
echo "=== K-e. CARE-KV error-decomposition figures ==="
python tools/make_carekv_before_after_3d.py \
  --model-id "${KE_MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}" \
  --out-dir "$PAPER_DIR/figures" \
  --layers ${KE_LAYERS:-0 11 21} \
  --seq-len "${KE_SEQ_LEN:-512}" \
  --max-tokens "${KE_MAX_TOKENS:-128}" \
  --max-channels "${KE_MAX_CHANNELS:-128}" \
  --plot-mode "${KE_PLOT_MODE:-all}" \
  --error-cmap "${KE_ERROR_CMAP:-inferno}" \
  --reduction-cmap "${KE_REDUCTION_CMAP:-RdBu_r}" \
  --residual-cmap "${KE_RESIDUAL_CMAP:-hot}" \
  --error-percentile "${KE_ERROR_PCT:-99}" \
  --overlay-top-percent "${KE_OVERLAY_PCT:-1.0}" \
  --base-bits "${KE_BASE_BITS:-3}" \
  --store-abs-k "${KE_STORE_ABS_K:-2}" \
  --store-abs-v "${KE_STORE_ABS_V:-4}" \
  --stats-json "$LOGS/Ke_error_stats.json" \
  2>&1 | tee "$LOGS/Ke_carekv_error_3d.log"

# ── M. Routing baseline ablation (optional, off by default) ─────
# Compares 6 residual-routing baselines (base_quant / random / magnitude_only
# / attention_only / carekv_score / oracle_proxy) under the same store + read
# budget. Default: synthetic prompt (~10 min). Set RUN_PHASE_M=1 to enable.
# Use M_DATASET=wikitext for the real WT-2 run (much slower).
if [ "${RUN_PHASE_M:-0}" = "1" ]; then
  echo "=== M. Routing baseline ablation ==="
  python tools/eval_routing_baselines.py \
    --out-csv "$PAPER_DIR/ablations/routing_baseline_ablation.csv" \
    --dataset "${M_DATASET:-synthetic}" \
    --seq-len "${M_SEQ_LEN:-64}" \
    --num-samples "${M_NUM_SAMPLES:-4}" \
    --base-bits "${M_BASE_BITS:-3}" \
    2>&1 | tee "$LOGS/M_routing_baseline.log"
  python tools/make_routing_baseline_figure.py \
    --csv "$PAPER_DIR/ablations/routing_baseline_ablation.csv" \
    --out "$PAPER_DIR/figures/fig_routing_baseline_ablation.png" \
    2>&1 | tee -a "$LOGS/M_routing_baseline.log"
fi

# ── N. Budget experiments (optional, off by default) ────────────
# Five sub-experiments (A1 ratio / A2 absolute / B store / C read /
# D K-V balance) sharing the same synthetic forward-pass eval. ~92 min
# total wall-clock on one GPU. Set RUN_PHASE_N=1 to enable.
if [ "${RUN_PHASE_N:-0}" = "1" ]; then
  echo "=== N. Budget experiments ==="
  python tools/eval_budget_experiments.py \
    --experiment "${N_EXPERIMENT:-all}" \
    --out-dir "$PAPER_DIR/ablations" \
    --seq-len "${N_SEQ_LEN:-64}" \
    --base-bits "${N_BASE_BITS:-3}" \
    2>&1 | tee "$LOGS/N_budget_experiments.log"
  python tools/make_budget_figures.py \
    --ablations-dir "$PAPER_DIR/ablations" \
    --figures-dir   "$PAPER_DIR/figures" \
    2>&1 | tee -a "$LOGS/N_budget_experiments.log"
fi

# ── O. Adaptive read-budget experiment (optional, off by default) ──
# Compares fixed RK=RV={1,2,3,4} vs adaptive_score with max RK=RV=4 and
# relative threshold in {0.00, 0.05, 0.10, 0.20, 0.30}. ~20 min for 9
# cells on the synthetic prompt. Set RUN_PHASE_O=1 to enable.
if [ "${RUN_PHASE_O:-0}" = "1" ]; then
  echo "=== O. Adaptive read-budget experiment ==="
  python tools/eval_adaptive_read_budget.py \
    --out-csv "$PAPER_DIR/ablations/adaptive_read_budget.csv" \
    --seq-len "${O_SEQ_LEN:-64}" \
    --base-bits "${O_BASE_BITS:-3}" \
    2>&1 | tee "$LOGS/O_adaptive_read_budget.log"
  python tools/make_adaptive_read_budget_figure.py \
    --csv "$PAPER_DIR/ablations/adaptive_read_budget.csv" \
    --out "$PAPER_DIR/figures/fig_adaptive_read_budget.png" \
    2>&1 | tee -a "$LOGS/O_adaptive_read_budget.log"
fi

# ── P-direct. Same-condition SOTA comparison (optional, off by default) ─
# Direct comparison of CARE-KV with same-condition baseline
# reimplementations (FP16, base_quant ladder, KIVI-style, CARE-KV fixed +
# adaptive). Stub adapters for KVQuant/MiKV/ZipCache emit "unsupported"
# rows with blocker text. Synthetic SL=64 ~16 min; WT-2 N=4 SL=128 ~3 h.
# Set RUN_PHASE_P_DIRECT=1 to enable.
if [ "${RUN_PHASE_P_DIRECT:-0}" = "1" ]; then
  echo "=== P-direct. Same-condition SOTA comparison ==="
  PD_TAG="synthetic_sl${PD_SEQ_LEN:-64}"
  if [ "${PD_DATASET:-synthetic}" = "wikitext" ]; then
    PD_TAG="wikitext2_n${PD_NUM_SAMPLES:-4}_sl${PD_SEQ_LEN:-128}"
  fi
  mkdir -p "$PAPER_DIR/sota_direct"
  python tools/eval_sota_direct_comparison.py \
    --out-csv  "$PAPER_DIR/sota_direct/sota_direct_${PD_TAG}.csv" \
    --out-json "$PAPER_DIR/sota_direct/sota_direct_${PD_TAG}.json" \
    --dataset "${PD_DATASET:-synthetic}" \
    --seq-len "${PD_SEQ_LEN:-64}" \
    --num-samples "${PD_NUM_SAMPLES:-4}" \
    2>&1 | tee "$LOGS/P_direct_${PD_TAG}.log"
  python tools/make_sota_direct_figures.py \
    --csv         "$PAPER_DIR/sota_direct/sota_direct_${PD_TAG}.csv" \
    --ppl-out     "$PAPER_DIR/figures/fig_sota_direct_ppl.png" \
    --mem-out     "$PAPER_DIR/figures/fig_sota_direct_memory_quality.png" \
    --runtime-out "$PAPER_DIR/figures/fig_sota_direct_runtime.png" \
    2>&1 | tee -a "$LOGS/P_direct_${PD_TAG}.log"
fi

# ── Q. CARE-KV on top of base quantizers (optional, off by default) ─
# Same-condition direct comparison of CARE-KV's residual correction on
# top of two base-quantizer schemes (uniform per-group vs KIVI-style
# per-channel K / per-token V). Phase Q-stacked is now end-to-end real
# (KIVI+CARE-KV rows are no longer stubs — see
# summaries/carekv_on_base_quantizers.md "Integration delivered").
# Set RUN_PHASE_Q=1 to enable. Knobs: Q_DATASET / Q_SEQ_LEN / Q_NUM_SAMPLES.
if [ "${RUN_PHASE_Q:-0}" = "1" ]; then
  echo "=== Q. CARE-KV on base quantizers ==="
  mkdir -p "$PAPER_DIR/ablations"
  python tools/eval_carekv_on_base_quantizers.py \
    --out-csv "$PAPER_DIR/ablations/carekv_on_base_quantizers.csv" \
    --dataset "${Q_DATASET:-wikitext}" \
    --seq-len "${Q_SEQ_LEN:-128}" \
    --num-samples "${Q_NUM_SAMPLES:-4}" \
    2>&1 | tee "$LOGS/Q_carekv_on_base_quantizers.log"
  python tools/make_carekv_on_base_quantizers_figure.py \
    --csv     "$PAPER_DIR/ablations/carekv_on_base_quantizers.csv" \
    --ppl-out "$PAPER_DIR/figures/fig_carekv_on_base_quantizers_ppl.png" \
    --mem-out "$PAPER_DIR/figures/fig_carekv_on_base_quantizers_memory_quality.png" \
    2>&1 | tee -a "$LOGS/Q_carekv_on_base_quantizers.log"
fi

# ── L. Summarize (regenerates final_report.md + artifact_list.txt) ──
echo "=== L. Summarize ==="
python tools/summarize_all_results.py "$PAPER_DIR" 2>&1 | tee "$LOGS/L_summarize.log"

echo
echo "============================================================"
echo "  paper_eval directory : $PAPER_DIR"
echo "  final report         : $PAPER_DIR/final_report.md"
echo "  summary tables       : $PAPER_DIR/summaries/"
echo "  figures              : $PAPER_DIR/figures/"
echo "  artifact list        : $PAPER_DIR/artifact_list.txt"
echo "============================================================"
