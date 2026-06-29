# VPhase A — CARE-KV runtime breakdown (BEFORE vectorization)

> **Headline**: On TinyLlama WT-2 SL=128 N=1, a single CARE-KV forward takes
> **415.8 s**, of which **98.3%** is the CARE-KV prefill correction and
> **82.7% is ROUTER SCORING alone** — `ResidualRouter.route` is called
> **90,112 times** (= Hq×T×L = 32×128×22), one per `(query_head, token,
> layer)`, at ~3.8 ms/call. The router per-`(h,t)` Python call is THE
> bottleneck; vectorizing it (batched scoring + `torch.topk`) is the
> highest-leverage change.

## Measured breakdown

| stage | time | % of forward | calls | ms/call |
|---|---|---|---|---|
| model_forward_total | 415.80 s | 100.0% | 1 | — |
| carekv_prefill | 408.76 s | 98.31% | 22 (layers) | 18 580 |
| sparse_correction_driver | 406.08 s | 97.66% | 22 | 18 458 |
| **router_scoring** | **343.78 s** | **82.68%** | **90 112** | **3.815** |
| correction apply (K+V) + overhead | ~62 s | ~15% | 90 112 | ~0.7 |

`PPL=11.77, K_reads=154 108, V_reads=206 340, peak=2700 MB.`
CSV: `ablations/carekv_runtime_breakdown_before.csv`.

(Profiler note: `apply_slot_corrections` is bound into `layer.py` at import,
so the module-level timer recorded 0 calls for it; its time shows up in the
~15% "correction apply + overhead" remainder. `router.route` is a class
method, so it was intercepted correctly — its 82.7% / 90 112 calls is the
robust signal.)

## What this means for the vectorization plan

- **Python-loop iteration count = 90 112** per forward (the `for h in
  range(Hq): for t in range(T):` driver loop at `layer.py:491-535`, summed
  over 22 layers). Each iteration calls `router.route()` once and
  `apply_slot_corrections()` once.
- **Arithmetic is trivial** (Part E: <0.3% of attention FLOP). The 415 s is
  pure Python/per-slot overhead — confirming the optimization is a
  loop-removal problem, not a math problem.
- **Priority (data-driven, refines the B→C→D plan):**
  1. **Router scoring (VPhase D) is the dominant cost (82.7%)** — batch the
     K/V candidate scoring across all `(head, token)` queries per kv_head and
     select with `torch.topk` instead of 90 112 per-query Python calls.
  2. **K correction (VPhase C)** — the per-`(h,t)` `apply_slot_corrections`
     K loop is the bulk of the remaining ~15%; batch `q·R_K` and the
     `(V−O_base)`-weighted accumulation with einsum/bmm.
  3. **V correction (VPhase B)** — already vectorized for the
     non-joint path (`vectorized_v_correction`); extend to the joint+both
     setting so the whole path avoids the per-`(h,t)` loop.
- The three are **coupled** under `policy=joint, kind=both`: the joint
  normalized top-k merges K and V candidates, so the batched router (D) and
  batched K/V apply (B,C) must be implemented together as one vectorized
  joint+both path to remove the loop entirely (this is exactly why the
  existing `correction_impl=vectorized` falls back to `cached` for
  joint+both — `layer.py:454-455`).

## Target

Replace the 90 112-call per-`(h,t)` router+correction loop with a per-kv_head
batched path: build slot-index tensors once, score all queries' candidates in
one shot, joint `torch.topk`, and apply K+V corrections via einsum/bmm —
preserving the math (PPL Δ ≤ 1e-4), the READ=0 early-return invariant, the
K_reads/V_reads counters, and GQA `repeat_kv` semantics.
