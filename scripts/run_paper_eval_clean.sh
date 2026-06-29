#!/usr/bin/env bash
#
# scripts/run_paper_eval_clean.sh
# --------------------------------
# Clean, paper-oriented CARE-KV rerun orchestrator.  See
# tools/paper_eval.py for the actual experiments and
# tools/summarize_paper_results.py for the final report.
#
# Environment overrides:
#   PAPER_DIR     — output dir (default: results/paper_eval_<ts>)
#   SEQ_LEN       — sequence length for PPL/sweeps (default: 128)
#   ARCHIVE_DIR   — where to move pre-existing result files
#   SKIP_HEAVY    — set to 1 to skip the slow sweep/ablation
#   SKIP_FIGURES  — set to 1 to skip matplotlib figures
#   TRY_USE_CACHE — set to 1 to also attempt USE_CACHE=1 generation (Phase 4)
#
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

TS=$(date +%Y%m%d_%H%M%S)
PAPER_DIR="${PAPER_DIR:-results/paper_eval_${TS}}"
ARCHIVE_DIR="${ARCHIVE_DIR:-results/archive_before_paper_eval_${TS}}"
SEQ_LEN="${SEQ_LEN:-128}"
SKIP_HEAVY="${SKIP_HEAVY:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"
TRY_USE_CACHE="${TRY_USE_CACHE:-0}"

mkdir -p "$ARCHIVE_DIR"
mkdir -p "$PAPER_DIR"/{logs,ppl,sweeps,ablations,memory,latency,figures,generation,summaries}

# Archive pre-existing top-level result files (not directories).
find results/ -maxdepth 1 -type f -exec mv {} "$ARCHIVE_DIR/" \; 2>/dev/null || true

# Preserve memory audit if present.
if [ -f "$ARCHIVE_DIR/memory_optimization_audit.md" ]; then
  cp "$ARCHIVE_DIR/memory_optimization_audit.md" "$PAPER_DIR/memory/"
fi

LOGS="$PAPER_DIR/logs"

# ── A. compile + tests ──────────────────────────────────────────
echo "=== A. compile + tests ==="
for f in __init__.py cache.py quantizer.py residual_store.py residual_router.py attention.py layer.py llama_patch.py utils.py tests/test_carekv_v2.py; do
  python -m py_compile $f && echo "OK $f"
done | tee "$LOGS/A_compile.log"

python tests/test_carekv_v2.py 2>&1 | tee "$LOGS/A_tests.log" || {
  echo "TESTS FAILED — stopping" >&2
  exit 1
}

# ── B. core PPL ─────────────────────────────────────────────────
echo "=== B. core PPL (SEQ_LEN=$SEQ_LEN) ==="
python tools/paper_eval.py core-ppl \
  --seq-len "$SEQ_LEN" \
  --out-csv "$PAPER_DIR/ppl/core_ppl.csv" \
  2>&1 | tee "$LOGS/B_core_ppl.log"

# ── C. invariant ────────────────────────────────────────────────
echo "=== C. invariant check ==="
python tools/paper_eval.py invariant \
  --seq-len "$SEQ_LEN" \
  --out-csv "$PAPER_DIR/ppl/invariant.csv" \
  2>&1 | tee "$LOGS/C_invariant.log"

# ── F. memory (fast, always runs) ───────────────────────────────
echo "=== F. memory table ==="
python tools/paper_eval.py memory \
  --out-csv "$PAPER_DIR/memory/memory_table.csv" \
  2>&1 | tee "$LOGS/F_memory.log"

# ── G. generation sanity ────────────────────────────────────────
echo "=== G. generation sanity ==="
GEN_ARGS=()
if [ "$TRY_USE_CACHE" = "1" ]; then GEN_ARGS+=(--try-use-cache); fi
python tools/paper_eval.py generation \
  --out-dir "$PAPER_DIR/generation" "${GEN_ARGS[@]}" \
  2>&1 | tee "$LOGS/G_generation.log"

# ── H. latency placeholder ──────────────────────────────────────
if [ "$TRY_USE_CACHE" = "1" ]; then
  echo "=== H. latency (Phase 4 / use_cache=True pending) ==="
  cat > "$PAPER_DIR/latency/README_phase4_pending.md" <<EOF
# Latency benchmark

Pending HF \`use_cache=True\` / DynamicCache integration (Phase 4 / Phase G).
Re-run \`scripts/run_paper_eval_clean.sh TRY_USE_CACHE=1\` once that lands.
EOF
else
  cat > "$PAPER_DIR/latency/README_phase4_pending.md" <<EOF
# Latency benchmark

Pending HF \`use_cache=True\` / DynamicCache integration (Phase 4 / Phase G).
Re-run with \`TRY_USE_CACHE=1\` once Phase G is implemented.
EOF
fi

# ── D. budget sweep (heavy) ─────────────────────────────────────
if [ "$SKIP_HEAVY" = "0" ]; then
  echo "=== D. stored-slot budget sweep ==="
  python tools/paper_eval.py budget-sweep \
    --seq-len "$SEQ_LEN" --bits-only-3 \
    --out-csv "$PAPER_DIR/sweeps/stored_budget_sweep.csv" \
    2>&1 | tee "$LOGS/D_budget_sweep.log"
else
  echo "=== D. budget sweep skipped (SKIP_HEAVY=1) ==="
fi

# ── E. V/K/both ablation (heavy) ────────────────────────────────
if [ "$SKIP_HEAVY" = "0" ]; then
  echo "=== E. V/K/both ablation (INT3) ==="
  python tools/paper_eval.py vk-ablation \
    --seq-len "$SEQ_LEN" \
    --out-csv "$PAPER_DIR/ablations/v_k_both_ablation_int3.csv" \
    2>&1 | tee "$LOGS/E_vk_ablation.log"
else
  echo "=== E. ablation skipped (SKIP_HEAVY=1) ==="
fi

# ── I. figures ──────────────────────────────────────────────────
if [ "$SKIP_FIGURES" = "0" ]; then
  echo "=== I. diagnostic figures ==="
  python tools/paper_eval.py figures \
    --out-dir "$PAPER_DIR/figures" \
    2>&1 | tee "$LOGS/I_figures.log" || echo "(figures step warned; continuing)"
fi

# ── J. summarize ────────────────────────────────────────────────
echo "=== J. summarize ==="
python tools/summarize_paper_results.py "$PAPER_DIR" 2>&1 | tee "$LOGS/J_summarize.log"

echo
echo "============================================================"
echo "  output dir       : $PAPER_DIR"
echo "  final report     : $PAPER_DIR/final_report.md"
echo "  summary tables   : $PAPER_DIR/summaries/"
echo "  artifact list    : $PAPER_DIR/artifact_list.txt"
echo "============================================================"
