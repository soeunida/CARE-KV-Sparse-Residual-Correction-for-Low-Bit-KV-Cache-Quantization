# Router BW pre-filter v2 — directional **sign-sketch** proxy (design)

> ### ⚠️ OUTCOME: implemented + tested — NOT adopted (magnitude bound wins).
> This design was motivated by a **NO-GO for the magnitude bound that turned out
> to be a normalization bug** (see `prefilter_sweep_results.md` correction). Once
> the bug was fixed, the plain magnitude bound `‖q‖·‖R_K‖` already holds PPL at
> C=12.5% (0.000 Δ at SL1024). The sign proxy below was built and swept
> (`CAREKV_ROUTER_SIGN_PREFILTER_B`, `prefilter_fix_sign32_*.csv`) and is **not
> better** — the magnitude bound is a *safe upper bound* (never drops a true
> winner: `‖q‖‖R_K‖ ≥ |q·R_K|`), while the sign estimate adds Hamming noise
> (Δ wobbles ±0.04…0.19). The design is retained for the record and reasoning;
> the shipped GO path is the simpler magnitude bound. — *The rest of this doc is
> the original design as written before the bug was found.*

> Follow-up to the (then-believed) NO-GO magnitude pre-filter. That bound
> (`|q·R_K| ≤ ‖q‖·‖R_K‖`) *appeared* to fail because `‖q‖` is per-query-constant,
> so it ranks by `‖R_K‖` and drops the directional term `|q·R_K|` — the reasoning
> below builds a **1-bit-per-dim sign sketch** (SimHash) to recover the *angle*
> between `q` and `R_K`. (In practice the magnitude ranking was fine; the earlier
> failure was the -inf normalization bug, not lost direction.)

## 0. What the exact router actually computes (grounded)

Per K sub-block `cg` (k_channel_group=32 dims), the store keeps
(`residual_store.py:281-282`, `416-430`):

```
rk_mean            = mean_t R_K[t, cg]              # (32,)
k_sketch[cg]       = rk_mean @ P        (fp16, 32-D)   # P = get_sketch_proj, (32,32)/√32
k_error_norm[cg]   = mean_t ‖R_K[t, cg]‖ (fp16, 1×2 B)
```

At decode the router forms `q_sketch = q_cg @ P` and scores with the **directional**
factor `qdotr = |q_sketch · k_sketch| ≈ |q · rk_mean|`. Reading `k_sketch` costs
**sketch_dim·2 = 64 B / candidate** — this is the dominant O(S) term
(`carekv_overhead_analysis.md`).

## 1. Insight — the sign of the sketch encodes the angle

Write `a = q_sketch`, `b = k_sketch` (both in R³²). The exact factor is
`|a·b| = ‖a‖·‖b‖·|cos θ|`. The magnitude bound kept `‖a‖‖b‖` and threw away
`|cos θ|` — exactly the missing direction. **SimHash** recovers `θ` from *signs*:
for the shared random `P`, the per-dim signs `sign(a)`, `sign(b) ∈ {±1}³²` satisfy

```
E[ Hamming(sign(a), sign(b)) / D ]  =  θ / π          # D = #sign bits
⇒  cos θ̂  =  cos( π · Hamming / D )
```

So a **b-bit sign code** of `k_sketch` (default b = sketch_dim = 32 bits = **4 B**)
plus the already-stored `k_error_norm` (a `‖b‖` proxy, 2 B) gives a **directional**
estimate

```
d̂  =  ‖q_sketch‖ · ‖b‖ · |cos(π·Hamming/D)|   ≈   |q · rk_mean|
```

at **6 B/candidate** instead of 64 B, and — unlike the magnitude bound — the
`|cos θ̂|` factor **promotes aligned slots even when `‖R_K‖` is small**, which is
precisely the case the magnitude filter buried.

## 2. Stored state — reuse the sketch's own sign plane (no new randomness)

Store, per K sub-block, the **1-bit sign plane of the existing sketch**:

```
k_sign[cg] = packbits( k_sketch[cg] > 0 )     # 32 bits = 4 B  (fp16 sketch → its MSB)
```

- **No new projection** — `k_sign` is a bit-plane of `k_sketch` (same `P`), so the
  proxy is a *consistent* coarsening of the exact estimator (tighter than an
  independent random hyperplane set).
- `k_error_norm[cg]` already exists → the `‖b‖` factor is free.
- New store cost: **+4 B/candidate** (a 1-bit plane); memory increase is
  `4 / slot_bytes ≈ <1%` of the residual slot. Optional: skip storing and read
  the sketch's sign bits directly if the layout allows MSB-first reads.

## 3. Two-stage algorithm (sign-first progressive read)

Per (layer, kv_head), per decode query `q`, per `cg`:

- **Stage 0 — attention gate (free).** Drop pages with `page_attn_mass < ε`
  (a_base already in memory). Sparse long-context attention removes most far
  pages at zero read.
- **Stage 1 — directional rank (6 B/candidate).** Read `k_sign` (4 B) + `k_error_norm`
  (2 B). Compute `q_sign = sign(q_cg @ P)` **once per query**, then per candidate
  `h = popcount(k_sign ⊕ q_sign)`, `cosθ̂ = cos(πh/D)`, and

  ```
  s_proxy = page_attn_mass · ‖q_sketch‖ · k_error_norm · |cosθ̂|
                            · boundary · v_diff · sens
  ```

  Keep the **top-C** by `s_proxy`.
- **Stage 2 — exact score (64 B × C).** Read the full `k_sketch` **only** for the
  C shortlisted, compute exact `|q_sketch·k_sketch|`, pick top-RK.

V side unchanged (already scalar-scored). Two knobs: **b** (sign bits read,
1…32 — progressive/MSB-first) and **C** (shortlist size).

## 4. Bandwidth model (per decode step, K-side scoring read)

```
current  (full sketch):  n_pages·SK · (D·2)               = 64  B/cand      (O(S))
sign-proxy Stage 1:      n_pages·SK · (b/8 + 2)            ≈ 6   B/cand (b=32) (O(S))
         + Stage 2:      C · (D·2)                         = O(1)
```

For b=32 the O(S) term drops **64 → 6 B ≈ 10.7×**; for b=8, **64 → 3 B ≈ 21×**
(coarser angle). Plugging into the overhead model, the ~8–10% BW-over-INT3
(dominated by this read) is expected to fall toward **~2–3%** *if C stays small*
— which is the open question the experiment answers. Net vs fp16 improves from
~0.22× toward the ~0.19× INT3 floor.

## 5. Correctness / confidence knob

SimHash is an **estimator**, so small-C pruning is approximate — but *directional*,
so it should hold PPL where the magnitude bound could not. Two regimes:

- **Approximate (default).** Sweep C at fixed b; expect Δ→0 at C ≪ pool (the
  magnitude bound only reached Δ=0 at C=pool).
- **Provable-ish.** From the binomial variance of `h` (σ² ≈ D·θ/π·(1−θ/π)), take a
  one-sided upper bound `cosθ_UB = |cos(π·(h − z·σ_h)/D)|` for the rank key →
  pruning that is exact with prob ≥ Φ(z), at the cost of a larger C. Report both.

## 6. Why this was *expected* to fix the NO-GO (original reasoning — see caveat)

> **Post-hoc:** this whole section rests on a NO-GO that was actually a
> normalization bug. The "structural failure" below never happened in a correct
> run — with the bug fixed, the magnitude ranking already selects the right
> slots. Kept for the record; the premise is false.

The (apparent) failure looked structural: aligned slots (`cos θ ≈ 1`) with small
`‖R_K‖` seemed to be ranked below orthogonal slots (`cos θ ≈ 0`) with large
`‖R_K‖`, because the key omitted `cos θ`. The sign proxy **puts `|cos θ̂|` back
into the key**, so an aligned small-norm slot with `‖R_K‖·|cosθ|` large would rank
correctly. In reality the remaining effect is only *angle-estimate noise* at small
`b` — and since there was no systematic mis-direction to correct, that noise makes
the sign proxy *worse* than the plain bound, as the sweep confirmed.

## 7. Implementation plan (localized, behind flags)

- `residual_store.py` (build): after computing `k_sketch`, store
  `k_sign = packbits(k_sketch > 0)` in `PageMeta` (new field, 4 B/cand). Gate on
  `CAREKV_ROUTER_SIGN_PREFILTER` so default layout is unchanged.
- `residual_router.py` `route` (cached path) and `attention.py`
  `vectorized_joint_correction` (fast path): replace the v1 magnitude UB with the
  sign proxy — Stage 1 uses `k_sign`+`k_error_norm`+`q_sign`; Stage 2 reads
  `k_sketch` for the top-C. Reuse the existing `_use_pf` / mask scaffolding.
- Flags: `CAREKV_ROUTER_SIGN_PREFILTER_B` (bits, default 32),
  `CAREKV_ROUTER_PREFILTER_C` (shortlist, default 0 = exact). C=0 ⇒ exact,
  invariant + paper-best unchanged (same gate as v1). Debug counters
  `k_prefilter_pool` / `k_prefilter_scored` + a new `k_sign_bits_read`.

## 8. Experiment plan

1. **Exactness gate:** C=0 (or C≥pool) bit-identical to the exact router; pytest
   invariants pass. (Same as v1.)
2. **(b × C) grid at long context (SL512, SL1024; the regime where O(S) matters):**
   b ∈ {8,16,32}, C ∈ {8,16,32,…,pool}. Metric: WikiText-2 PPL vs exact +
   `scored/pool` + analytical Stage-1 bytes. **Success bar:** some (b, C) with
   C ≪ pool holds ΔPPL ≲ 0.05 while cutting the O(S) sketch read ≥5–10× —
   i.e. Δ→0 *before* 100% kept (the curve v1 never achieved).
3. **Ablate the norm factor:** proxy with vs without `k_error_norm` — isolates how
   much the *direction* alone buys (the whole point vs v1).
4. **Attention-gate ε sweep** (Stage 0) — free candidate survival vs PPL.
5. **7B / SL≥1024** confirm on Mistral; update `carekv_overhead_analysis.md` with
   the sign-proxied O(S) term.

**GO:** a directional (b,C) holds PPL within noise at C ≪ pool → the O(S) read
drops ~5–20× and the overhead falls toward ~2–3%. **NO-GO:** if even the
direction-aware proxy needs C≈pool, the *magnitude* AND *angle* are both
insufficient and the full 32-D sketch is genuinely irreducible for ranking — then
the only levers left are the free attention-gate and a smaller `sketch_dim`.

## 9. Risks / notes

- **Angle noise at small b.** b=8 → σ(θ) ≈ 20° — may need larger C; b=32 is the
  tight end (still 10.7× BW cut). The b sweep quantifies it.
- **`k_error_norm` is mean per-token norm, not `‖rk_mean‖`.** Proportional enough
  for ranking; if it matters, store `‖k_sketch‖` (2 B) as the `‖b‖` factor instead.
- **Store cost.** +4 B/candidate (<1% of a residual slot); no per-token compute at
  store time beyond a sign+packbits.
- This targets **K** (V is scalar-scored already); if V ever moves to a sketch,
  the same sign proxy applies.

**Status: design.** Paper-best config unchanged until the (b, C) grid clears the
GO bar in §8. Supersedes the magnitude pre-filter for the BW-reduction goal;
the magnitude UB is retained only as the exact-but-useless baseline in the sweep.
