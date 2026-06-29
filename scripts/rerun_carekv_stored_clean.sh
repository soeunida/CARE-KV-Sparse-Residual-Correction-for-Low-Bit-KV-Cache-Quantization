#!/usr/bin/env bash
#
# scripts/rerun_carekv_stored_clean.sh
# -------------------------------------
# After run_paper_eval_clean.sh finished, rerun ONLY the carekv_stored-heavy
# pieces with the router fix that prevents tiny-positive read_budget from
# silently rounding to zero reads.  Reuses the existing paper_eval directory.
#
# Env:
#   PAPER_DIR  — existing paper_eval_<ts> dir to update (required)
#   SEQ_LEN    — sequence length (default 128)
#
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

: "${PAPER_DIR:?PAPER_DIR must be set to an existing results/paper_eval_<ts> dir}"
SEQ_LEN="${SEQ_LEN:-128}"
[ -d "$PAPER_DIR" ] || { echo "PAPER_DIR $PAPER_DIR does not exist" >&2; exit 1; }

LOGS="$PAPER_DIR/logs"

echo "=== rerun B. core PPL (router fix) ==="
python tools/paper_eval.py core-ppl \
  --seq-len "$SEQ_LEN" \
  --out-csv "$PAPER_DIR/ppl/core_ppl.csv" \
  2>&1 | tee "$LOGS/B_core_ppl_rerun.log"

echo "=== rerun D. budget sweep (router fix) ==="
python tools/paper_eval.py budget-sweep \
  --seq-len "$SEQ_LEN" --bits-only-3 \
  --out-csv "$PAPER_DIR/sweeps/stored_budget_sweep.csv" \
  2>&1 | tee "$LOGS/D_budget_sweep_rerun.log"

echo "=== rerun E. V/K/both ablation (router fix) ==="
python tools/paper_eval.py vk-ablation \
  --seq-len "$SEQ_LEN" \
  --out-csv "$PAPER_DIR/ablations/v_k_both_ablation_int3.csv" \
  2>&1 | tee "$LOGS/E_vk_ablation_rerun.log"

echo "=== Phase D. memory-pareto sweep ==="
python tools/paper_eval.py memory-pareto \
  --seq-len "$SEQ_LEN" \
  --out-csv "$PAPER_DIR/sweeps/memory_pareto_sweep.csv" \
  2>&1 | tee "$LOGS/D_memory_pareto.log"

echo "=== regenerate summary ==="
python tools/summarize_paper_results.py "$PAPER_DIR" 2>&1 | tee "$LOGS/J_summarize_rerun.log"

echo
echo "============================================================"
echo "  rerun done.  PAPER_DIR=$PAPER_DIR"
echo "  final report: $PAPER_DIR/final_report.md"
echo "============================================================"
