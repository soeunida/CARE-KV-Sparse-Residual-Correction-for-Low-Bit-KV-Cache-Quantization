# Router-bottleneck diagnostic — results + actions

TinyLlama, WikiText-2 PPL, INT3, paper-best CARE-KV. Tool:
`tools/diag_router_bottleneck.py`.

## #1 (executed) — Vectorized correction was already implemented; ~15–28× faster
`correction_impl=vectorized` (P5 `vectorized_joint_correction`, handles
joint+both) is **bit-close to cached and dramatically faster**. The earlier
slowness was only because the diagnostic hard-coded `cached`.

| arm (SL=64 N=2) | PPL | time |
|---|---:|---:|
| carekv **vec** sk16 | 17.301 | **25.4 s** |
| carekv **cached** sk16 | 17.201 | 390.5 s |

→ vec ≈ cached (Δ0.10 PPL at this tiny scale) at **~15× (small) – ~28× (larger)**
speed. **Action: use `vectorized` for heavy/iterative evals** (keep `cached` as
the bit-exact reference). This unblocks 7B validation and fast router iteration —
the same carekv eval dropped from 703 s → 25 s.

## Sketch-dim sweep (the router-scoring test) — higher-N is decisive
N=2 SL=64 was non-monotonic (noise). At **N=8 SL=128** (1008 tokens) it is
monotonic:

| sketch_dim | PPL (N=8 SL=128) |
|---|---:|
| 16 | 14.978 |
| **32** | **14.928** |
| 64 | 14.808 |
(fp16 13.543, base_int3 18.645.)

**Interpretation.** The K-residual channel group is 32-wide, so `sketch_dim=16`
is a **lossy 32→16 projection** of the residual used for the `|q·R_K|` ranking;
`sketch_dim=32` is **full-rank (no projection loss)**. The 16→32 gain is the
removal of that projection noise; the further 32→64 (−0.12) is over-complete
redundancy / noise. So K-scoring at sk16 was mildly noisy but **not the main
bottleneck** — sketch only moves PPL by ~0.05–0.17, while the residual gap to
fp16 is ~1.3. The remaining gap is **representation/budget**, not scoring.

## #2 (executed) — sketch_dim default 16 → 32 (full-rank)
Changed `CacheConfig.sketch_dim` 16→32 (+ adapter / eval_ppl_dataset kw).
Principled (= k_channel_group → exact `|q·R_K|` ranking), nearly free (sketch is
a small memory component), validated small gain. Verified default now 32.

## Takeaways for the roadmap
- **System (vectorization) is the dominant win and is already in-tree** → adopt
  it for big runs; it is the enabler for everything else.
- **Router K-scoring is near-adequate** (sketch full-rank gives only a small
  gain) → further router-scoring work has diminishing returns.
- The residual **+1.3 PPL gap to fp16 is representation/budget**-bound; prior
  work showed finer granularity rejected and reads>2 add noise, so this is the
  hard frontier (not cheaply closed).
- **Next: 7B validation** of these characteristics (now feasible at speed via
  the vectorized path).

**Status: diagnostic** (TinyLlama; vec≈cached validated; sketch at N=8 SL=128).
CSVs: `router_bottleneck_vec.csv`, `sketch_sweep_n8_sl128.csv`.
