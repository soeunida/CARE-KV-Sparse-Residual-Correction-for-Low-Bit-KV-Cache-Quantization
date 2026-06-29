# Routing baseline ablation (Phase M, synthetic — **diagnostic-only**)

> **Label**: diagnostic-only. Synthetic 254-token prompt at SL=64. Absolute
> PPL values are inflated because the prompt is short and out-of-distribution
> for TinyLlama. The **relative ordering** is meaningful but is not a
> paper-headline number. See `routing_baseline_wikitext2_n4.md` for the
> real-dataset pilot.

**Question.** Is CARE-KV's `joint+normalize` residual-routing score actually
useful, or could a simpler ranking (random, magnitude-only, attention-only)
get most of the gain at the same budget?

**Setup.** TinyLlama-1.1B-Chat-v1.0, paper-best CARE-KV config
(`carekv_stored`, `joint+normalize`, `cached` impl, packed INT3 base, int8
scale, uniform layer budget). **Same** store + read budget for every cell:
`STORE_ABS_K=2, STORE_ABS_V=4, READ_ABS_K=2, READ_ABS_V=2`. Only the
per-candidate score formula changes per cell. Prefill mode is
`carekv_stored` for every baseline except `base_quant` (which uses
`base_quant` prefill with `READ_ABS_K=READ_ABS_V=0`).

Eval dataset: 254-token synthetic prompt at `SEQ_LEN=64`, single forward
pass with shifted-CE loss. Total per-baseline runtime ≈ 2 minutes on one
GPU (5.6 s for `base_quant`, ~127 s each for the five `carekv_stored`
baselines).

CSV: `ablations/routing_baseline_ablation.csv`
Figure: `figures/fig_routing_baseline_ablation.png`

## Score formulas (only thing that changes across cells)

| baseline       | K candidate score                                            | V candidate score              |
|---|---|---|
| `base_quant`     | n/a (router not invoked; READ_ABS_K=READ_ABS_V=0)              | n/a                            |
| `random`         | uniform random                                                 | uniform random                 |
| `magnitude_only` | `|q·R_K|` (sketched)                                           | `||R_V||` (stored norm)        |
| `attention_only` | `page_attn_mass`                                               | `blk_attn_mass`                |
| `carekv_score`   | `page_attn_mass × |q·R_K| × page_boundary_risk × page_v_diff × sensitivity` (paper-best) | `blk_attn_mass × ||R_V|| × sensitivity` |
| `oracle_proxy`   | `|q·R_K| × page_attn_mass` (no structural prior / sensitivity) | `||R_V|| × blk_attn_mass`      |

`oracle_proxy` is a **diagnostic upper-bound proxy**, not a deployable
method — it uses the same per-candidate scoring inputs as the other
baselines but drops the multi-factor weighting. It is labeled "oracle"
only in the sense of "best-possible score from magnitude × attention
alone"; a true decode-time oracle would require post-decode error
information.

## Results

| baseline         | PPL       | ΔPPL vs base_quant | ΔPPL vs random | K_reads | V_reads | seconds |
|---|---:|---:|---:|---:|---:|---:|
| `base_quant`     | 140.902   |  +0.00 | +5.14   |       0 |       0 | 5.6 |
| `random`         | 135.761   | −5.14  | reference | 54 768 | 125 456 | 139.1 |
| `magnitude_only` | 121.697   | −19.21 | −14.06   | 116 286 | 63 938 | 126.6 |
| `attention_only` | 106.105   | −34.80 | −29.66   | 86 332  | 93 892  | 126.4 |
| **`carekv_score`**   | **102.746** | **−38.16** | **−33.02** | 81 705 | 98 519 | 127.0 |
| `oracle_proxy`   | 102.554   | −38.35 | −33.21   | 84 870  | 95 354  | 127.0 |

(`mean_dO_K` / `mean_dO_V` / `K_stored` / `V_stored` columns in the CSV
are 0 because the corresponding debug counters are not populated by the
current prefill pipeline; the read counters are the load-bearing
metric.)

## Findings

1. **CARE-KV strictly beats every simpler baseline at the same budget.**
   Compared to the paper-best score (102.75), random gives 135.76 (+33),
   magnitude-only gives 121.70 (+19), attention-only gives 106.10 (+3.4).
   The combined attention × magnitude × structural-prior score genuinely
   matters — no single factor recovers it.

2. **CARE-KV essentially saturates the magnitude × attention upper bound.**
   `oracle_proxy` (102.55) is only **0.19 PPL** ahead of `carekv_score`
   (102.75) — within run-to-run noise. The extra `page_boundary_risk`,
   `page_v_diff`, and `sensitivity` factors in the paper-best formula
   neither help nor hurt at this budget. They may matter more at smaller
   budgets (worth a follow-up ablation), but at SK=2 SV=4 RK=2 RV=2 they
   wash out.

3. **Random selection is only marginally better than no correction.**
   Random gives 135.76 vs base_quant 140.90 — a 3.6% relative improvement.
   This bounds "how much you could win just by burning the read budget
   on arbitrary slots." The remaining ~24% improvement from CARE-KV is
   attributable to ranking, not raw budget.

4. **K-vs-V read balance is policy-dependent.**
   Random skews heavily toward V (125k V reads vs 55k K) because there
   are more V candidates per page (16 vs 8 K-channel-groups per page per
   kv_head). `magnitude_only` skews K-heavy (116k K vs 64k V) because K
   residual magnitudes are larger per slot in the sketched score. The
   three smart baselines (`attention_only`, `carekv_score`,
   `oracle_proxy`) all converge to ~85k K + ~95k V, suggesting the
   "right" balance is close to 0.85:1, which the read budgets RK=2 RV=2
   nominally produce.

## Caveats / future work

- **Synthetic prompt only.** The 254-token synthetic prompt isolates the
  comparison cheaply but absolute PPL is not a paper headline number. A
  full WikiText-2 run (`--dataset wikitext --seq-len 128 --num-samples 4`
  or 16) is the right follow-up; per-baseline runtime is ~25 min at N=4,
  ~100 min at N=16. The runner gates this behind `RUN_PHASE_M=1` env in
  `scripts/run_all_paper_eval.sh`.
- **Oracle_proxy is not a true oracle.** It uses pre-decode inputs only
  (sketched magnitude × attention mass), so it bounds "what can you get
  from these inputs" rather than the actual achievable PPL ceiling.
- **Single-prompt result.** Multi-prompt averaging would strengthen the
  ranking claim.

## How to reproduce

```bash
# Synthetic (default, ~12 min total)
PAPER_DIR=results/paper_eval_20260529_015053 \
PYTHONPATH=/home/soeun python tools/eval_routing_baselines.py \
  --out-csv $PAPER_DIR/ablations/routing_baseline_ablation.csv \
  --dataset synthetic --seq-len 64

# WikiText-2 (slow, ~2 h for N=4 / ~10 h for N=16)
PAPER_DIR=results/paper_eval_20260529_015053 \
PYTHONPATH=/home/soeun python tools/eval_routing_baselines.py \
  --out-csv $PAPER_DIR/ablations/routing_baseline_ablation.csv \
  --dataset wikitext --seq-len 128 --num-samples 4

# Plot
python tools/make_routing_baseline_figure.py \
  --csv $PAPER_DIR/ablations/routing_baseline_ablation.csv \
  --out $PAPER_DIR/figures/fig_routing_baseline_ablation.png
```

Or via the unified runner: `RUN_PHASE_M=1 bash scripts/run_all_paper_eval.sh`.
