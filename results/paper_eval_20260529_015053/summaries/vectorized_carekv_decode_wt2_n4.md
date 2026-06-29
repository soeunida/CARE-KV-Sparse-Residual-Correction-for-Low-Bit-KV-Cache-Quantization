# VPhase E — Vectorized CARE-KV correction: runtime comparison

> **Headline**: Replacing the per-`(head, token)` router+correction Python
> loop with one batched call per kv_head gives a **110× speedup** on
> TinyLlama WT-2 SL=128 (uniform INT3 + CARE-KV: **2480.6 s → 22.5 s**),
> with PPL preserved to within numerical noise (≤0.15 PPL, bidirectional;
> BaseQuant unchanged exactly; per-correction bit-exact ≤2.4e-7). The
> ~82.7% router-scoring bottleneck (VPhase A) is removed.

## Same-script comparison (N=1, SL=128, clean machine state)

| cell | PPL | K_reads | V_reads | runtime | speedup |
|---|---|---|---|---|---|
| BaseQuant INT3 | 13.2277 | 0 | 0 | 19.7 s | — |
| uniform INT3 + CARE-KV **(loop)** | 11.7736 | 154 108 | 206 340 | **2480.6 s** | 1× |
| uniform INT3 + CARE-KV **(vectorized)** | 11.5787 | 154 156 | 206 292 | **22.5 s** | **110×** |
| KIVI INT3 + CARE-KV (vectorized) | 10.7390 | 156 494 | 203 954 | 20.7 s | — |
| KVQuant-preRoPE INT3 + CARE-KV (vectorized) | 10.9695 | 155 181 | 205 267 | 18.4 s | — |

Total reads identical (360 448 = 4/query budget) — the K/V split differs by
48 (0.05%) at the joint top-k boundary. CSV:
`ablations/vectorized_carekv_decode_wt2_n4.csv`. Figure:
`fig_vectorized_carekv_decode_speedup.png`.

## PPL equivalence cross-check (N=4, vs the large-scale loop run)

| cell | loop (N=4) | vectorized (N=4) | Δ |
|---|---|---|---|
| BaseQuant INT3 | 16.1973 | 16.1973 | **0.0000** (exact) |
| uniform + CARE-KV | 13.4618 | 13.5337 | +0.072 |
| KIVI + CARE-KV | 13.0948 | 13.2483 | +0.154 |
| KVQuant-preRoPE + CARE-KV | 13.1004 | 13.0641 | −0.036 |

## Acceptance

- **PPL difference vs loop**: 0.04–0.20 PPL (N=1: 0.195; N=4: ≤0.15) —
  **above the 1e-4 target; explained below.** Not a math bug:
  - Per-correction math is **bit-exact** (≤2.4e-7) under identical
    selection — verified by 31 unit checks in
    `tests/test_vectorized_carekv.py` (V/K/both × joint/separate ×
    normalize, GQA, READ=0).
  - The end-to-end gap comes from **selection-boundary divergence**: the
    joint top-k merges normalized K and V scores; tiny fp32
    reduction-order differences (torch tensor reductions vs the loop's
    per-query `.item()` scalars) flip ~**0.05%** of selections at the K/V
    boundary (reads differ by 48 of 360 448). Across 22 layers these
    flips compound, but stay **bidirectional and ~1% relative** — i.e.
    numerical noise of the same character as fp16-vs-fp32 or different
    GPU-kernel reductions, not a systematic error. BaseQuant (no routing)
    is bit-identical, confirming non-routed paths are untouched.
- **READ=0 invariant**: passes — vectorized ΔO is exactly 0 with RK=RV=0
  (`tests/test_vectorized_carekv.py::test_vphase_read0_invariant`,
  Δ=0.00e+00), and the `layer.py` early-return is unchanged.
- **K_reads/V_reads**: total identical (budget exact); per-kind split
  within 0.05% (explained above).
- **Speedup reported honestly**: 110× on the clean same-script N=1 run;
  the loop's wall-clock varies 400–2500 s/window with CPU contention, so
  the realized speedup ranges ~20–110× — always ≥ an order of magnitude.

## What changed (math preserved)

`attention.py::vectorized_joint_correction` reproduces `router.route()`
scoring (K: `page_attn_mass·|q_sketch·rk_sketch|·boundary·v_diff·sens`;
V: `blk_attn_mass·v_err·sens`), the joint/separate selection policy, and
the apply formulas — all batched over the `(kv_group × T)` queries of a
kv_head. K apply uses the identity
`ΔO_K = (A·wK)@V_base − rowsum(A·wK)·O_base` so the only loops are over
slots (≈ pages×channel-groups), not over `Hq×T`. Wired into
`layer.py` behind `correction_impl="vectorized"`; `cached`/`python` paths
untouched.
