"""tools/eval_budget_granularity_sweep.py

Phase N+ — residual-GRANULARITY sensitivity sweep (DIAGNOSTIC).

Motivation
----------
The store-budget sweep (experiment B) saturates because the per-page residual
candidate pools are fixed by granularity, not by the budget knob:

    K candidate cap = head_dim      / k_channel_group
    V candidate cap = ceil(page_size / v_token_block)

For the paper config (head_dim=64, k_channel_group=32, page_size=16,
v_token_block=4) that is K_cap=2, V_cap=4 — exactly the paper-best
(SK=2, SV=4).  So `SK=2, SV=4` is the *minimum-storage equivalent under the
current residual granularity*, NOT a proven globally optimal store budget.

This sweep varies the granularity itself to change the candidate caps and see
whether finer residuals (more candidates, more memory) actually lower PPL:

    k_channel_group ∈ {64, 32, 16}   → K_cap ∈ {1, 2, 4}
    v_token_block   ∈ {8,  4,  2}    → V_cap ∈ {2, 4, 8}

Each cell stores ALL candidates at its granularity (SK=K_cap, SV=V_cap) — the
minimum-storage-equivalent point for that granularity — and reads at the paper
budget (RK=min(2,K_cap), RV=min(2,V_cap)).  We report PPL, reads, recovered
sparsity, residual memory, and runtime so the PPL-vs-memory trade can be read
directly.

Small evaluation (per user spec):
    TinyLlama-1.1B, SEQ_LEN=128, BASE_BITS=3,
    route_policy=joint, score_normalize=1, correction_impl=cached.

Outputs:
    <out-dir>/budget_granularity_sweep.csv
  (figure + markdown are produced by make_budget_figures / a companion step)

DIAGNOSTIC: this is a small synthetic-prompt forward-pass sweep, not a
paper-scale dataset run.  It validates the candidate-cap interpretation; it
does NOT by itself establish a new paper-best granularity.
"""
from __future__ import annotations
import argparse, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # tools/ on path
import eval_budget_experiments as B   # reuse run_cell / _write_csv / _abs_env


K_CHANNEL_GROUPS = [64, 32, 16]   # → K_cap = head_dim / kcg (head_dim=64)
V_TOKEN_BLOCKS   = [8, 4, 2]      # → V_cap = ceil(page_size / vtb) (page_size=16)
HEAD_DIM = 64
PAGE_SIZE = 16


def _cells():
    cells = []
    for kcg in K_CHANNEL_GROUPS:
        k_cap = HEAD_DIM // kcg
        for vtb in V_TOKEN_BLOCKS:
            v_cap = math.ceil(PAGE_SIZE / vtb)
            # store ALL candidates at this granularity (minimum-storage
            # equivalent for the granularity); read at the paper budget,
            # clamped to availability.
            sk, sv = k_cap, v_cap
            rk, rv = min(2, k_cap), min(2, v_cap)
            env = B._abs_env(sk, sv, rk, rv)
            env["CAREKV_K_CHANNEL_GROUP"] = str(kcg)
            env["CAREKV_V_TOKEN_BLOCK"]   = str(vtb)
            label = f"gran_kcg{kcg}_vtb{vtb}"
            cells.append((label, env))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--base-bits", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "budget_granularity_sweep.csv")
    print(f"\n===== budget granularity sweep (SL={args.seq_len}, INT{args.base_bits}) =====",
          flush=True)
    rows = []
    for label, env in _cells():
        try:
            r = B.run_cell(label, env, args.seq_len, args.base_bits)
            rows.append(r)
            print(f"[gran] {label:22s} "
                  f"Kcap={r['K_cand_cap']} Vcap={r['V_cand_cap']} "
                  f"effSK={r['eff_SK']} effSV={r['eff_SV']}  "
                  f"PPL={r['ppl']:.4f}  "
                  f"K_reads={r['K_reads']:>8d} V_reads={r['V_reads']:>8d}  "
                  f"recK={r['recovered_K_elems']:>10d} recV={r['recovered_V_elems']:>10d}  "
                  f"resMB={r['residual_mem_MB']:.2f}  ({r['seconds']:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f"[gran] {label} ERROR: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(label=label, error=str(e)))
    B._write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
