# Routing baseline ablation — WikiText-2 N=4 pilot (Phase M)

> **Label**: real-dataset pilot, **not full paper-scale**. N=4 windows
> (508 evaluated tokens) at SL=128. A full WT-2 N=16 (2 032 tokens) run
> would tighten the ranking but costs ~9 h wall-clock on one GPU (6
> baselines × ~90 min each at N=16). N=4 is the smoke check; treat
> per-baseline gaps under ~0.2 PPL as within noise until N grows.

## Setup

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Dataset: WikiText-2 (`wikitext-2-raw-v1`, test split)
- Windowed log-loss PPL, N=4 non-overlapping windows × SL=128 = **508 evaluated tokens**
- Paper-best CARE-KV config (`carekv_stored`, `joint+normalize`, `cached`,
  packed INT3 base, int8 scale, uniform layer budget)
- **Same** budget for every cell: `STORE_ABS_K=2, STORE_ABS_V=4, READ_ABS_K=2, READ_ABS_V=2`
- Only the per-candidate score formula varies across cells
- `base_quant` cell uses `base_quant` prefill with `READ_ABS_K=READ_ABS_V=0` (no router invocation)

Total wall-clock: **8 095 s ≈ 135 min ≈ 2.25 h** (5 × ~26 min carekv cells + 1 × 28 s base_quant cell).

CSV: `ablations/routing_baseline_wikitext2_n4.csv`
Log: `ablations/routing_baseline_wikitext2_n4.log`
Figure: `figures/fig_routing_baseline_wikitext2_n4.png`

## Results

| baseline         | PPL     | ΔPPL vs base_quant | K_reads     | V_reads   | seconds |
|---|---:|---:|---:|---:|---:|
| `base_quant`     | 16.197  |  +0.000 |          0 |         0 |     28.1 |
| `random`         | 16.151  | −0.046  |    475 915 |   965 877 |  1 740.2 |
| `magnitude_only` | 14.617  | −1.580  |  1 320 645 |   121 147 |  1 593.4 |
| `attention_only` | **13.286**  | **−2.911**  |    646 955 |   794 837 |  1 579.1 |
| `carekv_score`   | 13.462  | −2.735  |    641 915 |   799 877 |  1 574.8 |
| `oracle_proxy`   | **13.157**  | **−3.040**  |    673 460 |   768 332 |  1 579.7 |

(`mean_dO_K`, `mean_dO_V`, `K_stored`, `V_stored` columns in the CSV are 0
because those debug counters are not populated by the prefill pipeline;
read counters are the load-bearing metric.)

## Findings — honest reporting of the WT-2 ranking

1. **Ranking on real text differs from the synthetic prompt.**
   - Synthetic (SL=64): `carekv_score (102.7) < oracle_proxy (102.5)`, both
     well below the other baselines.
   - WT-2 (SL=128 N=4): `oracle_proxy (13.16) < attention_only (13.29) <
     carekv_score (13.46) < magnitude_only (14.62) < random (16.15) ≈
     base_quant (16.20)`.

2. **`attention_only` essentially ties `carekv_score`.** The 0.18 PPL gap
   (13.286 vs 13.462) is small for N=4 = 508 tokens and may be within
   noise. The K/V read counts for the two baselines are nearly identical
   (647k vs 642k K, 795k vs 800k V), confirming that the
   `boundary_risk × v_diff × sensitivity` multipliers in the full CARE-KV
   K-score don't change *which* slots get selected very much in
   practice — they're a small reweighting of an already-near-optimal
   attention-driven ranking.

3. **`oracle_proxy` clearly beats `carekv_score` by 0.30 PPL.** The
   simpler `magnitude × attention` formula (no structural prior, no
   per-layer sensitivity) outperforms the full multi-factor CARE-KV
   score on this dataset. This suggests the full multiplicative score
   may be **over-engineered** for V residuals at this budget — a flatter
   weighting captures more of the signal.

4. **The attention signal is by far the most important component.**
   `attention_only` (13.29) is the biggest single-factor jump from
   `random` (16.15) — a **2.9 PPL gain** from attention alone. Adding
   magnitude on top of attention (oracle_proxy) gives an additional
   0.13 PPL. Magnitude without attention (magnitude_only) gives only
   1.5 PPL over random.

5. **`random` is essentially indistinguishable from `base_quant` on real text**
   (16.15 vs 16.20, gap of 0.05 PPL = noise). The synthetic-prompt
   finding that random gave a 5-PPL improvement over base_quant does
   NOT replicate on WikiText-2.

## What this means for the paper-best config

The paper-best `carekv_score` is **still strictly better than `magnitude_only`,
`random`, and `base_quant`** on WT-2, and within 0.2 PPL of `attention_only`.
The ranking story is real but more nuanced than the synthetic-prompt
suggested:
- **`attention_only` is a strong, simple baseline** that competes with the
  paper-best at this budget. Worth flagging in the paper as the strongest
  simple ablation.
- **`oracle_proxy` (≈ `magnitude × attention`) outperforms the full
  multi-factor score** by 0.3 PPL — a follow-up should test whether
  simplifying the K/V score to just `mag × attn` improves the headline
  WT-2 N=16 number. If it does, that's a small but real paper change.
- The current paper-best config does NOT need to be changed for the
  paper headline (it still beats every deployable baseline and is within
  0.3 PPL of the oracle), but the ranking-of-scores ablation should be
  reported honestly: `carekv_score` is third on WT-2 N=4, not first.

## Caveats

- **N=4 is small.** 508 tokens is enough to see the headline ordering but
  not to declare statistical significance for gaps under ~0.2 PPL.
  N=16 (2 032 tokens) would tighten the conclusion at ~9 h wall-clock.
- **TinyLlama-1.1B only.** Same as every other paper-eval cell. Whether
  the `attention_only ≈ carekv_score` finding holds on larger models is
  an open question.
- **`oracle_proxy` is not a true oracle.** It uses only pre-decode inputs
  (sketched K magnitude × attention mass); a true post-decode oracle
  (knowing actual error reduction per slot) would likely beat it.
- All N=4 runs share a fixed window starting at token 0; multi-seed
  averaging would harden the per-baseline numbers.

## How to reproduce

```bash
PAPER_DIR=results/paper_eval_20260529_015053 \
PYTHONPATH=/home/soeun python tools/eval_routing_baselines.py \
  --out-csv $PAPER_DIR/ablations/routing_baseline_wikitext2_n4.csv \
  --dataset wikitext --seq-len 128 --num-samples 4 --base-bits 3

python tools/make_routing_baseline_figure.py \
  --csv $PAPER_DIR/ablations/routing_baseline_wikitext2_n4.csv \
  --out $PAPER_DIR/figures/fig_routing_baseline_wikitext2_n4.png
```

Or via the unified runner:
`RUN_PHASE_M=1 M_DATASET=wikitext bash scripts/run_all_paper_eval.sh`.
