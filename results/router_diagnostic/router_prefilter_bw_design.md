# Router O(S) scoring-BW optimization — bound-based pre-filter (design)

> Follows the overhead analysis (`carekv_overhead_analysis.md` / REBUTTAL §1):
> CARE-KV's correction adds ~8–10% read-bandwidth over INT3, **dominated by the
> router's O(S) sketch-scoring read** — each decode step reads a `sketch_dim`-wide
> fp16 sketch (≈64 B) for *every* stored K candidate (≈ n_pages·SK of them). The
> residual apply itself is O(1). This design cuts that O(S) sketch read without
> changing which slots get corrected.

## 0. Where the bytes go (current router)

Per decode step, per (layer, kv_head), `ResidualRouter.route` walks **all** pages
and, for **every** stored K candidate, reads `k_sketch` (sketch_dim·2 B) to
compute `|q·R_K|`, then ranks and keeps top-RK. V candidates are already scored
by a **scalar** `v_error_norm` (2 B) — cheap. So the dominant O(S) traffic is the
**K sketches**: `n_pages · SK · sketch_dim · 2` B.

## 1. Insight — a cheap scalar upper-bounds the expensive score

The exact K score uses `|q·R_K|` (needs the sketch). By Cauchy–Schwarz:

```
|q · R_K|  ≤  ‖q‖ · ‖R_K‖  =  ‖q‖ · k_error_norm
```

`k_error_norm` is **already stored** per candidate (PageMeta, 1 fp16 scalar =
2 B), and `‖q‖` is a per-query constant. So

```
ub(candidate) = page_attn_mass · ‖q‖ · k_error_norm · boundary_risk · v_diff · sens
```

is a **provable upper bound** on the true K score (the sketch only sharpens the
`|q·R_K|` factor downward). → We can **prune** any candidate whose upper bound is
below the RK-th best *without reading its sketch*, and the result is **exact**.

## 2. Design — two-stage (+ free attention gate)

Per (layer, kv_head), per decode query:

- **Stage 0 — attention gate (free).** `a_base` is already in memory (base
  attention). Skip pages with `page_attn_mass < ε` (or keep the top-M attended
  pages). No extra read; in long context attention is sparse so most far pages
  drop out.
- **Stage 1 — cheap upper-bound rank (2 B/candidate).** For surviving
  candidates read only `k_error_norm` (2 B, **32× smaller** than the 64 B
  sketch) and compute `ub(candidate)`. Keep the top-**C** by `ub`
  (C = α·RK, small).
- **Stage 2 — exact score (64 B × C).** Read `k_sketch` **only** for the C
  shortlisted candidates, compute exact `|q·R_K|`, pick top-RK.

V side is unchanged (already scalar-scored). Optionally apply the same gate.

**Correctness knob C:**
- **Exact** if `C ≥ #{candidates with ub ≥ (RK-th true score)}` — safe pruning,
  bit-identical to the current router.
- **Approximate** for small C: a candidate is only lost if its true score is
  top-RK but its upper bound ranked it below C — rare, and bounded by the
  ub–score gap. Sweep C to trade PPL vs BW.

## 3. Bandwidth model (per decode step, K-side scoring read)

```
current  :  n_pages · SK · (sketch_dim·2)                       # O(S), 64 B/cand
proposed :  n_pages · SK · 2      (error norms, O(S))
          + C · (sketch_dim·2)    (sketches, O(1))              # C ≪ n_pages·SK
```

For sketch_dim=32 the O(S) term drops **32×** (64 B → 2 B/candidate); the sketch
read becomes O(1). Plugging into the overhead model, the ~8–10% BW-overhead-over-
INT3 (dominated by this read) is expected to fall to **~1–1.5%** (residual O(1)
read + the tiny 2 B/candidate norm read), and the attention gate cuts it further
in sparse-attention long context. **CARE-KV read-BW vs fp16 improves accordingly
(from ~0.22× toward the ~0.19× INT3-base floor).**

## 4. Why it's safe / cheap to build

- `k_error_norm`, `k_sketch`, `v_error_norm` are **already stored** (PageMeta) —
  no new state, no extra store-time cost, no memory increase.
- Stage 0 uses `a_base` already computed for base attention.
- Change is localized to `ResidualRouter.route` (K-candidate loop): add the
  norm-based shortlist before the sketch dot-product.

## 5. Experiment plan

1. **Correctness/exactness:** at C=∞ (no prune) the router is bit-identical
   (READ=0 and paper invariants unaffected). Verify PPL unchanged vs current.
2. **C sweep (quality vs BW):** C ∈ {RK, 2·RK, 4·RK, 8·RK, ∞}, measure WikiText-2
   PPL (should be flat until C small) + the analytical sketch-read reduction +
   the measured `k_sketch` bytes read (add a counter).
3. **Attention-gate ε / top-M sweep:** measure candidate survival and PPL vs ε.
4. **Long-context (SL≥1024) on 7B:** the regime where O(S) matters — confirm the
   BW overhead drops from ~9% toward ~1% with PPL held (use the vectorized path).
5. **Update the overhead analysis** with the pre-filtered O(S) term.

**GO:** C small (≈2–4·RK) holds PPL within noise while cutting the O(S) sketch
read ~10–30×. **NO-GO / partial:** if PPL degrades at any C that saves BW, the
sketch is genuinely needed for ranking (then keep C conservative for the exact
regime — still a large BW win via the norm pre-read).

## 6. Risks / notes

- `‖q‖` per-query and `k_error_norm` give a *loose* upper bound (Cauchy–Schwarz);
  a loose bound → larger C for exactness. Tightening (e.g., a 1-D sign sketch, or
  storing `‖R_K‖` per sub-block) trades a little store memory for a tighter bound.
- The V side is already scalar-scored, so this targets K; if V ever moves to a
  sketch-scored scheme, the same bound applies.
- Compute (FLOPs) is already single-digit %; this is a **bandwidth** optimization
  — the axis that matters since decode is bandwidth-bound.

**Status: design → tested → GO for the magnitude bound.**
See `prefilter_sweep_results.md`. NOTE: an earlier revision recorded this as
NO-GO — that was a **normalization bug** (the shortlist masked `score_K` to `-inf`
*before* the `score_normalize` mean, corrupting every C<pool). After moving the
mask to `score_K_r` (post-normalize), the magnitude bound **holds PPL at C = 12.5%
of pool** (+0.046 at SL512, **0.000 at SL1024**) while cutting the O(S) sketch
read ~8×. The directional sign-sketch proxy (`router_sign_prefilter_design.md`)
was also implemented and tested but is **not better** — the magnitude bound is a
safe *upper bound* (never drops a true winner), whereas the sign estimate adds
Hamming noise. Paper-best config unchanged (pre-filter defaults to C=0 = exact).
