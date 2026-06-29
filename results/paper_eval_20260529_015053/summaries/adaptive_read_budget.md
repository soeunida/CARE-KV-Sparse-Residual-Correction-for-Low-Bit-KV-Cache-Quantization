# Adaptive read budget (Phase O — synthetic, **diagnostic-only**)

> **Status**: prototype-evaluation, **diagnostic-only**. Synthetic
> 254-token prompt at SL=64. The WT-2 N=4 confirmation has since landed
> with a different optimal threshold (rel=0.05 on WT-2 vs rel=0.10
> here); see `adaptive_read_budget_wikitext2_n4.md`. Both sweeps agree
> on the headline that the adaptive filter improves over fixed
> `RK=RV=2`, but the optimal threshold is **dataset-dependent**.

## Motivation

The Phase N budget sweep (B/C tables) showed that **fixed read budget is
non-monotonic** — more reads are not always better:

| (RK, RV) | PPL    | reads (K + V)     |
|---|---:|---:|
| (1, 1)   | 107.80 |  43 k +  47 k     |
| **(2, 2)** | **102.75** |  82 k +  99 k |
| (3, 3)   | 108.52 | 103 k + 168 k     |
| (4, 4)   | 104.65 | 130 k + 230 k     |

Hypothesis: reading more residual slots brings in low-quality slots that
add noise on top of base attention. Treating `READ_ABS_K/V` as a **max**
rather than a **mandatory count**, and dropping slots whose score is
below a relative quality threshold, should beat fixed at the same max.

## Implementation

New config field (`cache.py`):

```python
read_budget_mode: str = "ratio"   # one of {"ratio", "absolute", "adaptive_score"}
read_relative_threshold: float = 0.0
read_absolute_threshold: float = 0.0
read_min_keep: int = 0
read_score_temperature: float = 1.0    # reserved (unused)
correction_norm_clip: float = 0.0      # config-only, no enforcement yet
```

New env vars (defaults preserve paper-best):

```
CAREKV_READ_BUDGET_MODE=adaptive_score    # opt in
CAREKV_READ_RELATIVE_THRESHOLD=0.10       # keep slot iff score >= 0.10 * top_score
CAREKV_READ_ABSOLUTE_THRESHOLD=0.0
CAREKV_READ_MIN_KEEP=0                    # floor on slot count when candidates exist
CAREKV_READ_SCORE_TEMPERATURE=1.0         # reserved
CAREKV_CORRECTION_NORM_CLIP=0.0           # config-only, no enforcement
```

Behavior in `residual_router.py` (`route()`):

- `_resolve_read_budgets` treats `"adaptive_score"` like `"absolute"` for
  the **max cap** — `READ_ABS_K/V` are upper bounds.
- After the existing policy dispatch (joint / separate / k_first /
  adaptive) selects up to `(budget_k, budget_v)` slots, an **adaptive
  filter** drops slots whose score is below
  `max(rel_threshold × top_score, abs_threshold)`. If `min_keep > 0` and
  the filter would leave fewer slots, the filter floors at `min_keep`.
- New debug counters: `router_requested_RK/V`, `router_effective_RK/V_sum`,
  `router_skipped_{K,V}_by_{relative,absolute}_threshold`,
  `router_n_route_calls`.

**Paper-best path is byte-identical**: `read_budget_mode="absolute"` (the
existing default for the paper config) skips the new filter entirely;
the `route_policies_and_absolute_budgets` unit test still passes.

## Results — Phase O CSV (`ablations/adaptive_read_budget.csv`)

Setup: TinyLlama-1.1B, INT3, paper-best store budget (`SK=2, SV=4`),
synthetic 254-token prompt at SL=64.

| label                                | mode             | rel | PPL      | requested (RK, RV) | effective (RK, RV) | K_reads | V_reads | skipped K (rel) | skipped V (rel) |
|---|---|---:|---:|---|---|---:|---:|---:|---:|
| fixed_RK1_RV1                          | absolute       | —   | 107.803  | (1.00, 1.00) | (0.96, 1.04) | 43 408   |  46 704 | — | — |
| **fixed_RK2_RV2**                      | absolute       | —   | **102.746** | (2.00, 2.00) | (1.81, 2.19) | 81 705   |  98 519 | — | — | **paper-best**
| fixed_RK3_RV3                          | absolute       | —   | 108.523  | (3.00, 3.00) | (2.28, 3.72) | 102 806  | 167 530 | — | — |
| fixed_RK4_RV4                          | absolute       | —   | 104.654  | (4.00, 4.00) | (2.89, 5.11) | 130 377  | 230 071 | — | — |
| adaptive_maxRK4_maxRV4_rel0.00         | adaptive_score | 0.00| 104.654  | (4.00, 4.00) | (2.89, 5.11) | 130 377  | 230 071 | 0       | 0       | sanity — identical to fixed_RK4_RV4
| adaptive_maxRK4_maxRV4_rel0.05         | adaptive_score | 0.05| 112.524  | (4.00, 4.00) | (1.98, 3.19) |  89 148  | 143 846 | 41 523  | 85 931  | over-filters (worse)
| **adaptive_maxRK4_maxRV4_rel0.10**     | adaptive_score | 0.10| **101.902** | (4.00, 4.00) | (1.88, 2.65) |  84 553  | 119 312 | 45 674  | 110 909 | **best — beats paper-best by 0.84 PPL**
| adaptive_maxRK4_maxRV4_rel0.20         | adaptive_score | 0.20| 105.708  | (4.00, 4.00) | (1.75, 2.06) |  78 936  |  92 893 | 51 525  | 137 094 |
| adaptive_maxRK4_maxRV4_rel0.30         | adaptive_score | 0.30| 104.060  | (4.00, 4.00) | (1.63, 1.79) |  73 627  |  80 458 | 56 947  | 149 416 |

(`mean_dO_K` / `mean_dO_V` columns are 0 in the CSV — those counters
aren't populated by the prefill pipeline, same caveat as the routing
baseline ablation.)

## Findings

1. **Adaptive `rel=0.10` strictly beats fixed `RK=RV=2`** by **0.84 PPL**
   (101.90 vs 102.75) — a real, measurable improvement on this prompt.
   Effective reads (1.88 K, 2.65 V) are close to the fixed budget but
   slightly V-heavier; total K reads are ~3 % more, V reads ~21 % more.
   The router uses the extra V budget when the top-V scores warrant it
   and skips when they don't.

2. **Threshold sweet spot is narrow.** `rel=0.05` is too lax (112.5 —
   worse than fixed `RK=RV=2`); `rel=0.20` and `rel=0.30` are too strict
   (105.7 and 104.1). Only `rel=0.10` improves.

3. **Sanity check holds**: `adaptive rel=0.00 == fixed_RK4_RV4`
   (both PPL 104.654, identical reads). Threshold=0 produces zero
   filtering — the "no-op invariant" is preserved.

4. **READ=0 invariant is preserved** at the config layer:
   `read_budget_mode="adaptive_score"` with `READ_ABS_K=READ_ABS_V=0`
   shares the same `_resolve_read_budgets` path as `"absolute"`, which
   returns `(0, 0)` immediately. The `route()` function's adaptive-mode
   block only runs when `selected` is non-empty.

5. **Skipped counts are consistent with the threshold**. Going from
   rel=0.05 to rel=0.30, `skipped_V_by_relative_threshold` rises
   85 931 → 149 416 (~1.7× growth) and effective_RV_mean drops 3.19 →
   1.79 (~1.8× drop). The filter is doing what it advertises.

## Implications for paper-best config

- **Replacing `fixed RK=RV=2` with `adaptive_score max RK=RV=4 rel=0.10`
  is a real read-budget improvement on this prompt** (−0.84 PPL).
- **Caveat**: this is on a single 254-token synthetic prompt and is
  prototype-evaluation, not paper-ready. The right confirmation is to
  run the same sweep on WikiText-2 N=4 SL=128 (~2.3 h wall-clock) and
  verify the ordering survives. If WT-2 ranks the same, the paper-best
  config should be updated.
- **For now**: keep the paper-best at `fixed RK=RV=2` (per the
  no-silent-change rule) and document `adaptive_score rel=0.10` as the
  improvement candidate pending the WT-2 confirmation run.

## Caveats / future work

- **Single synthetic prompt.** Absolute PPL numbers are inflated; the
  relative ordering is what matters.
- **Reduction-panel sweet spot may be model-/budget-dependent.** The
  rel=0.10 finding may not transfer to different store budgets, different
  models, or longer contexts. Add a multi-config sweep before paper headline.
- **`correction_norm_clip` is config-only** (no enforcement yet) — only
  needed if adaptive thresholding hadn't been enough; it was, so this
  stays as a future safety guard.
- **`read_score_temperature` is reserved** for a future softmax-style
  probabilistic gate; currently a no-op.

## How to reproduce

```bash
PYTHONPATH=/home/soeun python tools/eval_adaptive_read_budget.py \
  --out-csv results/paper_eval_20260529_015053/ablations/adaptive_read_budget.csv \
  --seq-len 64 --base-bits 3

python tools/make_adaptive_read_budget_figure.py \
  --csv results/paper_eval_20260529_015053/ablations/adaptive_read_budget.csv \
  --out results/paper_eval_20260529_015053/figures/fig_adaptive_read_budget.png
```

Or via the runner: `RUN_PHASE_O=1 bash scripts/run_all_paper_eval.sh`.
