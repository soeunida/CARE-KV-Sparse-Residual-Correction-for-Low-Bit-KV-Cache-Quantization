# VPhase F — Long-context feasibility re-test (post-vectorization)

> **Headline**: The 110× vectorization makes long-context CARE-KV PPL
> **feasible**. On TinyLlama, vectorized CARE-KV (uniform INT3) now runs at
> **SL=512 in 63.8 s** and **SL=1024 in 131.8 s** (single window) — both
> were ~tens of GPU-hours with the Python loop. CARE-KV **improves
> BaseQuant INT3** at long context (SL=512: 8.87 → 8.23, −0.65; SL=1024:
> 6.40 → 5.81, −0.58), with nonzero K/V reads. Real Qwen2.5-7B fp16/BaseQuant
> at SL=512 run in ~20 s (fp16 6.97, INT3 8.24).

## Results

| model | cell | SL | PPL | Δ vs BaseQuant | runtime | feasible |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | fp16 | 512 | 6.9728 | — | 20.8 s | yes |
| Qwen2.5-7B | BaseQuant INT3 | 512 | 8.2405 | — | 12.2 s | yes |
| TinyLlama | BaseQuant INT3 | 512 | 8.8728 | — | 38.5 s | yes |
| TinyLlama | **uniform + CARE-KV (vec)** | 512 | **8.2258** | **−0.6471** | 63.8 s | yes |
| TinyLlama | BaseQuant INT3 | 1024 | 6.3985 | — | 69.0 s | yes |
| TinyLlama | **uniform + CARE-KV (vec)** | 1024 | **5.8143** | **−0.5842** | 131.8 s | yes |

CARE-KV reads (SL=1024): K=1 228 497 V=1 655 087 (router fired). peak GPU
5.6 GB. CSV: `ablations/long_context_vectorized_carekv_feasibility.csv`.

## The reported questions

1. **Does vectorization make Qwen2.5-7B SL=512 feasible?** For fp16 +
   BaseQuant, **yes** (~12–21 s). CARE-KV itself cannot run on Qwen2.5-7B —
   the CARE-KV patch is **Llama-only** (`Qwen2ForCausalLM` unsupported), the
   same blocker as the large-scale Part B. So the CARE-KV long-context
   demonstration is on TinyLlama (supported Llama GQA); a true 7B CARE-KV
   run needs a Qwen2 patch port (separate work — vectorization removes the
   *speed* barrier, not the *architecture* barrier).

2. **Is SL=1024 N=1 feasible?** **Yes** — 131.8 s on TinyLlama (was ~tens of
   GPU-hours with the loop; the large-scale phase declared it infeasible).
   peak GPU 5.6 GB.

3. **Estimated runtime for SL=1024 N=4?** ≈ **4 × 132 s ≈ 9 min** (linear in
   windows), versus ~**1.5+ days** with the Python loop. The vectorized path
   makes paper-scale long-context evals routine.

4. **Does CARE-KV PPL improve BaseQuant INT3 at long context?** **Yes** —
   SL=512: −0.65 PPL (8.87 → 8.23); SL=1024: −0.58 PPL (6.40 → 5.81). The
   sparse-residual correction continues to recover quantization error as
   context grows; the absolute PPLs also fall with context (more tokens to
   attend to), and CARE-KV tracks below BaseQuant throughout.

5. **Remaining bottleneck?** Two:
   (a) **Architecture** — CARE-KV is Llama-only; 7B CARE-KV needs the Qwen2
   port. (b) **Memory of the batched scorer** — the V-diff term builds a
   `(Q, N, D)` intermediate (≈2 GB at SL=1024); for SL≫1024 or large batch
   this should be chunked over queries. Runtime is no longer the limiter
   (it is now dominated by the base model forward, not the correction).

## Caveat

Single-window (N=1) PPLs — directional, not paper-final. The point is
*feasibility* (runtime) and the *sign* of the CARE-KV improvement at long
context, both confirmed. Numerical note from VPhase E applies (vectorized
PPL differs from the loop by ~1% due to joint-boundary selection noise).
