# Adaptive read budget — WikiText-2 N=4 confirmation (Phase O)

> **Status**: real-dataset pilot, **not full paper-scale**. N=4 windows
> at SL=128 = 508 evaluated tokens. A WT-2 N=16 run (~10 h wall-clock
> for the 7-cell sweep) would tighten the conclusion. Treat per-cell PPL
> gaps under ~0.2 as within noise at this N.

## Setup

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Dataset: WikiText-2 (`wikitext-2-raw-v1`, test split), N=4 non-overlapping
  SL=128 windows → 508 evaluated tokens
- Paper-best CARE-KV config (carekv_stored, joint+normalize, cached,
  packed INT3, scale int8, uniform layer budget), `STORE_ABS_K=2 STORE_ABS_V=4`
- Only the read budget knobs vary per cell
- Total wall-clock: **10 016 s ≈ 167 min ≈ 2.8 h** (1 base_quant cell at
  27 s + 6 carekv cells × ~1 670 s each)

CSV: `ablations/adaptive_read_budget_wikitext2_n4.csv`
Log: `ablations/adaptive_read_budget_wikitext2_n4_run.log`
Figure: `figures/fig_adaptive_read_budget_wikitext2_n4.png`

## Results

| label                                  | mode             | rel | PPL      | eff (RK, RV) | K_reads    | V_reads    | skipped V (rel) | seconds |
|---|---|---:|---:|---|---:|---:|---:|---:|
| base_quant                             | absolute       | —   | 16.197   | (0.00, 0.00) |          0 |          0 |          0 |    27.4 |
| fixed_RK2_RV2                          | absolute       | —   | 13.462   | (1.78, 2.22) |    641 915 |    799 877 |          0 | 1 696.4 |
| fixed_RK4_RV4                          | absolute       | —   | 13.132   | (2.41, 5.59) |    869 425 |  2 014 159 |          0 | 1 712.9 |
| **adaptive_maxRK4_maxRV4_rel0.05**     | adaptive_score | 0.05| **12.932** | (1.93, 3.38) |    696 092 |  1 217 100 |    795 840 | 1 596.7 |
| adaptive_maxRK4_maxRV4_rel0.10         | adaptive_score | 0.10| 13.400   | (1.85, 2.83) |    667 770 |  1 019 933 |    994 238 | 1 691.2 |
| adaptive_maxRK4_maxRV4_rel0.20         | adaptive_score | 0.20| 13.589   | (1.70, 2.30) |    611 376 |    829 257 |  1 185 009 | 1 623.2 |
| adaptive_maxRK4_maxRV4_rel0.30         | adaptive_score | 0.30| 13.379   | (1.57, 2.00) |    565 681 |    719 910 |  1 293 605 | 1 669.3 |

(`mean_dO_K` / `mean_dO_V` columns are 0 in the CSV — those counters are
not populated by the current prefill pipeline; the K/V_reads and
effective_R*_mean counters are the load-bearing metrics.)

## Findings (honest reporting)

1. **adaptive `rel=0.05` is the WT-2 N=4 winner.**
   PPL **12.932** — beats fixed `RK=RV=2` by **−0.530 PPL** and fixed
   `RK=RV=4` by −0.200 PPL. The same budget cap (`max RK=RV=4`) gives a
   strictly better PPL with the filter than without.

2. **The threshold sweet spot shifted from the synthetic prompt.**
   - Synthetic (SL=64): `rel=0.10` won (101.90 vs fixed_RK2_RV2 = 102.75)
     and rel=0.05 was bad (112.52).
   - WT-2 (SL=128, N=4): `rel=0.05` wins (12.93) and rel=0.10 is *worse
     than* fixed_RK2_RV2 (13.40 vs 13.46).
   The optimal threshold is **dataset- and budget-dependent**, so any
   paper recommendation must include the dataset-specific tuning step.

3. **Non-monotonicity is dataset-dependent too.**
   On synthetic, fixed RK=RV=3 was strictly worse than RK=RV=2 (the
   originally-cited non-monotonicity). On WT-2, **fixed RK=RV=4 is
   actually better than RK=RV=2** (13.13 vs 13.46) — more reads help on
   real text up to a point. But the adaptive filter still beats the
   bigger fixed budget at fewer effective reads.

4. **Adaptive rel=0.05 is also more efficient than fixed RK=RV=4.**
   - Adaptive 0.05: 696 k K + 1 217 k V = 1.91 M total reads
   - Fixed RK=RV=4: 869 k K + 2 014 k V = 2.88 M total reads
   - **~34 % fewer reads** for slightly better PPL.

5. **Sanity gates passed.**
   - base_quant: 16.197 — matches the prior Phase M routing baseline
     run exactly (same N=4 windows, same paper-best config).
   - fixed_RK2_RV2: 13.462 — matches the prior Phase M `carekv_score`
     cell (13.462) exactly. The paper-best path is unchanged.
   - effective_R*_mean ≤ requested_R*_mean in every adaptive cell.
   - `adaptive rel=0.0` was not in this sweep but the synthetic run
     showed it == fixed_RK4_RV4 (sanity for the no-op invariant).

## Recommendation

**Recommend `read_budget_mode=adaptive_score` with `max RK=RV=4` and
`rel_threshold=0.05` as the candidate paper-best read-budget config on
WikiText-2.** The improvement over fixed `RK=RV=2` is **−0.53 PPL** on a
real dataset, and the improvement is consistent with the synthetic
ablation's broader claim that the filter improves over plain fixed
budgets (even though the optimal threshold value differs by dataset).

**Caveats before flipping the paper-best:**
- **N=4 is small.** A WT-2 N=16 run is the right confirmation.
- **Sweet spot is dataset-dependent.** rel=0.05 on WT-2 ≠ rel=0.10 on
  synthetic. The paper should describe the threshold as a tuned
  hyperparameter, not a universal constant. A small calibration sweep
  per dataset is the honest framing.
- **Only TinyLlama-1.1B was tested.** Multi-model confirmation pending.

**Do NOT flip the paper-best config silently.** Keep `fixed RK=RV=2` as
the default until a WT-2 N=16 (or larger) run reconfirms the ordering
and the sweet-spot calibration story is in the paper.

## How to reproduce

```bash
PYTHONPATH=/home/soeun python tools/eval_adaptive_read_budget.py \
  --out-csv results/paper_eval_20260529_015053/ablations/adaptive_read_budget_wikitext2_n4.csv \
  --dataset wikitext --seq-len 128 --num-samples 4 \
  --preset wikitext2_n4 --base-bits 3

python tools/make_adaptive_read_budget_figure.py \
  --csv results/paper_eval_20260529_015053/ablations/adaptive_read_budget_wikitext2_n4.csv \
  --out results/paper_eval_20260529_015053/figures/fig_adaptive_read_budget_wikitext2_n4.png
```

Or via the unified runner: `RUN_PHASE_O=1 O_DATASET=wikitext O_PRESET=wikitext2_n4 bash scripts/run_all_paper_eval.sh`.
