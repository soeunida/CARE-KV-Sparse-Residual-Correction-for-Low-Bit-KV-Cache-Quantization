# Router O(S) scoring-BW pre-filter — C sweep results

> ### ⚠️ CORRECTION (supersedes the earlier NO-GO conclusion)
> An earlier version of this doc reported the magnitude pre-filter as **NO-GO**
> (+1.2…+1.9 PPL at any BW-saving C). That was a **normalization bug**, not a real
> result: the pre-filter masked `score_K` to `-inf` *before* the `score_normalize`
> per-kind mean, so `score_K.abs().mean()` became `inf` and corrupted `score_K_r`
> (→ nan/0) for **every** C<pool, identically. Fixed in `attention.py`: the
> shortlist mask is now applied to `score_K_r` **after** normalization (score_K is
> left intact for the mean). **Post-fix verdict: GO for the magnitude bound.**
> pytest 20/20 still passes; C=0 / C≥pool remain bit-exact.

Config: TinyLlama-1.1B, `carekv_stored`, vectorized correction, sketch_dim=32,
paper-best SK2 SV4 RK2 RV2, WikiText-2 N=4. Read budgets are **global** per
(layer, kv_head) (`_resolve_read_budgets` absolute mode: bk=bv=2), so the joint
policy selects top-(2+2)=4 residual slots out of the full (n_k+n_v) pool — reads
are sparse, so a shortlist can safely restrict *which* K candidates are scored.
`CAREKV_ROUTER_PREFILTER_C=C` keeps only the top-C K slots per query (by a cheap
proxy) eligible for selection; C=0 or C≥pool = exact. `scored/pool` = the
analytical O(S) sketch-read fraction. `CAREKV_ROUTER_SIGN_PREFILTER_B` picks the
proxy: **0 = magnitude bound `‖q‖·‖R_K‖`** (Cauchy–Schwarz upper bound);
**b>0 = directional sign-sketch (SimHash) `‖a‖‖b‖·|cos θ̂|`** from b sign bits.

## Post-fix: magnitude bound (sign_b=0) — GO

| kept (scored/pool) | SL512 PPL | SL512 Δ | SL1024 PPL | SL1024 Δ |
|---:|---:|---:|---:|---:|
| 12.5% | 11.3685 | **+0.0456** | 10.4046 | **0.0000** |
| 25%   | 11.3276 | +0.0047 | 10.4046 | 0.0000 |
| 50%   | 11.3229 | 0.0000 | 10.4046 | 0.0000 |
| 100% (exact) | 11.3229 | 0.0000 | 10.4046 | 0.0000 |

- **Keeping just 12.5% of K candidates holds PPL** — +0.046 at SL512, **exactly
  0.000 at SL1024**. The O(S) sketch-scoring read (the ~8–10% BW-over-INT3 term)
  drops **~8×** with no quality loss.
- **Better at longer context.** SL1024 is bit-exact from 12.5% up; sparser
  attention concentrates the true winners into a smaller shortlist. This is the
  regime that matters for KV compression.

## Post-fix: directional sign-sketch proxy (sign_b=32) — not better

| kept | SL512 Δ | SL1024 Δ |
|---:|---:|---:|
| 12.5% | +0.1048 | +0.0410 |
| 25%   | +0.1386 | −0.1092 |
| 50%   | +0.1863 | −0.1725 |
| 100%  | 0.0000 | 0.0000 |

- The sign proxy is **noisier** than the plain magnitude bound (Δ wobbles
  ±0.04…0.19) and does **not** improve on it. The design intuition — "direction
  is the missing ingredient" — was drawn from the *buggy* NO-GO curve; once the
  bug is fixed, the magnitude bound already holds PPL, and the sign estimate's
  Hamming noise only *adds* variance.
- **Why magnitude wins.** `‖q‖·‖R_K‖ ≥ |q·R_K|` is a true **upper bound**, so a
  real winner (large `|q·R_K|`) always has a large bound and is **never dropped**
  from the shortlist. The sign estimate `|cos θ̂|` can *under*estimate a winner
  (quantization noise) and evict it. Safety (over-inclusion) beats a tighter but
  noisy point estimate for shortlisting.

Figure: `fig_prefilter_sweep.png` (post-fix magnitude vs sign, both SL).

## Bandwidth consequence (GO path = magnitude bound)

Two-stage K scoring per decode step (design §3):
```
Stage 1 (O(S)):  read k_error_norm (2 B) + page factors, rank by the bound
Stage 2 (O(1)):  read the full 32-D sketch (64 B) only for the top-C
```
At C = 12.5% of pool: the O(S) sketch read is replaced by a 2 B/candidate norm
read + a C-sized sketch read → the ~8–10% BW-over-INT3 (sketch-dominated) is
expected to fall toward **~2–3%**, holding PPL (SL1024: exactly). The full sign
plane is **not** needed; the already-stored `k_error_norm` scalar suffices.

## Files
- code: `attention.py` (`vectorized_joint_correction`: `pf_keep_k` post-norm
  mask + magnitude / sign proxy), `residual_router.py` (cached `route`).
  Flags `CAREKV_ROUTER_PREFILTER_C` (0=exact), `CAREKV_ROUTER_SIGN_PREFILTER_B`
  (0=magnitude bound).
- data (post-fix): `prefilter_fix_mag_sl512.csv`, `prefilter_fix_mag_sl1024.csv`,
  `prefilter_fix_sign32_sl512.csv`, `prefilter_fix_sign32_sl1024.csv`.
- data (pre-fix, superseded — bug artifact): `prefilter_sweep_sl512.csv`,
  `prefilter_sweep_sl1024.csv`, `prefilter_smoke_sl128.csv`.
- tools: `tools/eval_router_prefilter_sweep.py` (`--sign-b`),
  `tools/make_prefilter_figure.py`.

**Status: diagnostic — GO for the magnitude-bound pre-filter (paper-best
unchanged, defaults to C=0=exact).** Sign-sketch proxy implemented and tested;
not adopted (no gain over the simpler, safer bound).
