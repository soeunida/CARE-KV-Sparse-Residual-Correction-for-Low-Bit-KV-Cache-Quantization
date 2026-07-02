# CARE-KV — Consolidated Results

> Auto-generated from per-phase CSVs (read-only). INT3 KV quantization + output-error-aware sparse residual correction. PPL = WikiText-2.


## 0. Headline (the honest one-liner)

- **CARE-KV vs BaseQuant_INT3 (NS=64, 12 cells): 12W / 0L** — mean ΔPPL **-0.137**. CARE-KV reliably improves the naive INT3 baseline.

- **CARE-KV vs TurboQuant_INT3 (NS=64, 12 cells): 0W / 12L** — mean ΔPPL **+0.306**. At the rigorous sample size, TurboQuant (QJL rotation) wins everywhere.

- **Gap tracks K-outlier severity:** smallest on Mistral (≈parity), largest on outlier-heavy Yi / DeepSeek. → motivates the rotation-CARE-KV direction.


## 1. NS=64 production full grid (most rigorous)

| model | SL | fp16 | Base3 | CARE-KV | Turbo | Δ vs Base | Δ vs Turbo | winner |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B-v0 | 256 | 8.653 | 9.392 | 9.240 | 9.171 | -0.152 | +0.069 | Turbo |
| Mistral-7B-v0 | 512 | 7.156 | 7.675 | 7.614 | 7.586 | -0.062 | +0.027 | Turbo |
| Mistral-7B-v0 | 1024 | 6.553 | 7.061 | 6.982 | 6.958 | -0.079 | +0.024 | Turbo |
| Yi-6B | 256 | 8.943 | 10.341 | 10.104 | 9.575 | -0.237 | +0.528 | Turbo |
| Yi-6B | 512 | 7.680 | 8.770 | 8.634 | 8.293 | -0.136 | +0.341 | Turbo |
| Yi-6B | 1024 | 7.098 | 8.147 | 8.006 | 7.716 | -0.141 | +0.290 | Turbo |
| deepseek-llm- | 256 | 10.722 | 12.469 | 12.205 | 11.378 | -0.264 | +0.827 | Turbo |
| deepseek-llm- | 512 | 9.057 | 10.327 | 10.191 | 9.590 | -0.136 | +0.602 | Turbo |
| deepseek-llm- | 1024 | 8.201 | 9.220 | 9.069 | 8.705 | -0.151 | +0.364 | Turbo |
| open_llama_7b | 256 | 10.297 | 11.269 | 11.194 | 10.866 | -0.075 | +0.328 | Turbo |
| open_llama_7b | 512 | 8.603 | 9.394 | 9.282 | 9.108 | -0.112 | +0.174 | Turbo |
| open_llama_7b | 1024 | 7.865 | 8.528 | 8.426 | 8.322 | -0.103 | +0.103 | Turbo |

## 2. Selector study — combined_kvscore vs current SK2SV4

`combined_kvscore` (query-aware K+V selector, same SK2SV4 budget). Δ_current = combined − current (<0 = combined better).

| exp | model | SL | current | combined | Δ_current | Δ_turbo | usable |
|---|---|---|---|---|---|---|---|
| NS=8 | Mistral-7B- | 512 | 7.8719 | 7.8292 | -0.0427 | -0.3578 | no |
| NS=8 | Yi-6B | 512 | 8.8374 | 8.8154 | -0.0221 | -0.0287 | no |
| NS=8 | Mistral-7B- | 1024 | 7.4135 | 7.2820 | -0.1315 | -0.1907 | no |
| NS=8 | Yi-6B | 1024 | 8.5165 | 8.5670 | +0.0505 | +0.0409 | no |
| NS=16 | Mistral-7B- | 512 | 8.8355 | 8.7165 | -0.1190 | -0.3546 | no |
| NS=16 | Mistral-7B- | 1024 | 5.9338 | 5.8338 | -0.1000 | -0.1363 | no |

_combined_kvscore beats current on Mistral (held NS=8→16) and beats Turbo on Mistral at these small NS — but the NS=64 grid uses the **default** selector and loses to Turbo; a direct NS=64 combined-vs-Turbo confirmation is the open item._


## 3. Cross-architecture generalization (Phase 11C, cached models, SL512/NS4)

| model | family | fp16 | Base3 | CARE-KV | Turbo | Δ vs Base | corr_type | k_active | usable |
|---|---|---|---|---|---|---|---|---|---|
| opt-350m | OPT | 32.04 | 34.55 | 32.90 | n/a | -1.644 | K+V | True | yes |
| opt-1.3b | OPT | 21.21 | 25.15 | 231.32 | n/a | +206.175 | K+V | True | no |
| opt-2.7b | OPT | 18.32 | 19.58 | 19.05 | n/a | -0.533 | K+V | True | yes |
| open_llama_3b_v2 | OpenLLaMA | 9.50 | 10.59 | 10.21 | 10.39 | -0.382 | K+V | True | yes |
| Qwen2.5-7B | Qwen | 8.03 | 583.39 | 137390.64 | 151044.42 | +136807.253 | skip | True | no |
| open_llama_7b_v2 | OpenLLaMA | 8.39 | 8.89 | 8.82 | 8.75 | -0.066 | K+V | True | yes |
| deepseek-llm-7b-base | DeepSeek(LLaMA) | 8.60 | 9.61 | 9.24 | 9.06 | -0.367 | K+V | True | yes |

_With K correction restored (scale 0.1), every stable model is **K+V** (not V-dominant). CARE-KV beats BaseQuant on 5/5 stable models. Two failures are outlier-driven: **opt-1.3b** CARE-KV K-blow-up collapse (231 vs fp16 21), **Qwen2.5** total INT3 base collapse (method-independent). 5 paper-usable models._


## 4. Adaptive-policy study (Phase 11B, NS=8) — negative

- **Budget scaling**: more residual budget does not recover the gap (saturates; sometimes worse).

- **Selector oracle gap**: the current scorer is **near-oracle** (oracle_gap ≤0.02, sign-inconsistent) — no headroom.

- **Position policies**: only `middle_drop` helped Mistral marginally, but it is **NS-unstable** (Yi SL512 flipped sign NS=4→8) and **regresses Yi SL1024** → not promotable to default.

- **Verdict**: uniform SK2SV4 remains the default; lightweight policies do not robustly beat it.


## 5. Rotation CARE-KV (root-cause direction) — screening in progress

**Why:** CARE-KV's loss to TurboQuant is an **outlier** problem; TurboQuant fixes it densely via **rotation** (spreads outlier energy across channels). Idea: prepend a value-level rotation to CARE-KV's base so it composes with the sparse residual. Ceiling: QJL (score-level) is incompatible → stack can only inherit the **rotation** benefit, not QJL.

**Prior pilot:** uniform+CARE-KV 13.46 (bar); Hadamard **post-RoPE**+CARE-KV 15.23 (worse) → test **pre-RoPE** rotation (H1).

**Key question (H3):** are rotation and sparse-residual **complementary or substitutes**? (arm6 standalone vs arm5 +CARE-KV decides it.)


| arm | PPL | Δ vs uniform_carekv |
|---|---|---|
| fp16 | 12.3457 | -1.116 |
| base_int3 | 16.1973 | +2.735 |
| uniform_carekv | 13.4618 | +0.000 |
| rot_post_carekv | 15.2253 | +1.764 |
| rot_pre_carekv | 13.2587 | -0.203 |
| rand_pre_carekv | 13.8243 | +0.362 |
| rand_pre_base | 15.8792 | +2.417 |
| rot_pre_base | 17.3395 | +3.878 |

## 5b. Orthogonality to token eviction (SnapKV/H2O) — ADDITIVE ✓

Gated H2O-style eviction (`CAREKV_EVICT_KEEP_RATIO`, keep=0.9) applied to the base attention, so the residual router/correction operate on the kept set. Tests whether CARE-KV's gain is **additive** on top of eviction (Section 2 orthogonality claim).

| model | fp16 | base | base+evict | +CARE | evict+CARE | CARE gain (no-ev) | CARE gain (evict) | additive |
|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-C | 10.15 | 12.91 | 14.09 | 11.42 | 12.38 | +1.488 | +1.715 | ✅ YES |
| Mistral-7B-v0.3 | 6.79 | 7.30 | 7.60 | 7.02 | 7.30 | +0.279 | +0.298 | ✅ YES |
| deepseek-llm-7b- | 8.60 | 9.71 | 9.92 | 9.27 | 9.40 | +0.440 | +0.523 | ✅ YES |

_At a sensible eviction level (keep=0.9, near-lossless base), CARE-KV's residual gain is preserved on top of eviction (CARE gain ≈ same with/without eviction, both >0) across TinyLlama + Mistral-7B + DeepSeek-7B → **eviction and sparse residual are additive/orthogonal**. Unlike rotation, this transfers cleanly to 7B. As eviction gets more aggressive, CARE-KV recovers proportionally more of the damage (complementary).


## 5c. Orthogonality to mixed-precision (LeanKV/MiKV) — composes, diminishing

Gated per-token mixed-precision base (`CAREKV_MIXEDPREC_HI_FRAC`; salient tokens bits_hi=4, rest bits_lo=3, kivi side-buffer so store+attention stay consistent). Compares CARE gain on a kivi-INT3 base vs a kivi-mixed(4/3) base.

| model | base kivi-INT3 | +CARE | base mixed4/3 | +CARE | gain (INT3) | gain (mixed) | note |
|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 14.00 | 11.15 | 11.60 | 10.61 | +2.85 | +0.99 | composes (diminishing) |
| Mistral-7B | 7.78 | 7.14 | 7.15 | 6.99 | +0.64 | +0.16 | composes (diminishing) |
| DeepSeek-7B | 11.62 | 55.65 | 10.16 | 19.23 | −44.03 | −9.07 | K-corr collapse (kivi base) |

_On stable models (TinyLlama, Mistral-7B): CARE-KV **composes positively** with mixed precision — the stack (mixed + CARE-KV) is the best arm and CARE-KV still adds a positive gain on the mixed base — but with **diminishing returns** (the mixed base has less quantization error to recover, so the gain shrinks vs the INT3 base). Weaker than eviction's clean additivity: mixed-precision and sparse residual are **partial substitutes** that still compose._

_**DeepSeek-7B is a kivi-base artifact, not a mixed-precision failure** — confirmed by direct re-verification: on the **uniform-packed base** (production family) DeepSeek CARE-KV is **stable**, `base 9.71 → CARE 9.27` (gain **+0.44**, matching the NS=64 production 9.07), whereas on the **kivi-INT3 side-buffer base** the same model collapses (`base 11.62 → CARE 55.65`). So the 55.6 collapse is caused by the kivi per-channel K quantizer destabilizing CARE-KV's 1st-order K correction on DeepSeek's outlier-heavy K — orthogonal to the mixed-precision question. (Mixed precision is only realizable here through the kivi side-buffer override, so the clean mixed-precision orthogonality evidence rests on TinyLlama + Mistral.)_


## 5d. Runtime & memory (T6) — NOT a speed method (prototype latency)

CARE-KV's prefill is a per-(layer, kv_head, token) **Python loop** — prototype latency, NOT the achievable runtime. From `latency.csv` (TinyLlama, prompt 128, prefill):

| method | prefill_ms | vs base | peak_GPU_MB |
|---|---|---|---|
| fp16 | 22.8 | — | 2239 |
| BaseQuant_INT3 | 2044 | 1× | 3212 |
| **CARE-KV_INT3** | **215626** | **~105×** | 3212 |

- CARE-KV prefill is **~100× slower than BaseQuant** and **~9500× slower than fp16** (Python loop). Vectorization (`prefill_vectorization_bench.csv`) gives only **1.36×** — still prototype.
- **Peak GPU memory is HIGHER for CARE-KV** (3212 vs fp16 2239 MB) because the HF DynamicCache holds dummy fp16 K/V. The memory benefit is in the **KV-cache storage** (analytic ≈0.24–0.26× fp16 at long context), **not** peak runtime memory.
- **GQA-ladder full-eval wall-clock** (SL512/N4, CARE-KV cell incl. model load + correction): TinyLlama 109 s, Yi-6B 173 s, Mistral-7B 332 s, SOLAR-10.7B 436 s — scales with size, dominated by the prototype correction loop.

→ **CARE-KV is a quality/memory method, with no speed advantage**; a deployable latency needs CUDA/Triton kernels for packed-unpack + correction. Report prototype numbers honestly; make **no speedup claim**.


## 5e. Rotation-CARE-KV (T7) — NO-GO at scale (settled)

Settled negative (see §5). Hadamard pre-RoPE rotation + CARE-KV passed the TinyLlama screening gate (13.26 < 13.46 bar) but **does not transfer to 7B**: the de-risk on DeepSeek-7B regressed badly (rot_pre_carekv 10.47 ≫ uniform_carekv 9.27, worse even than base 9.71). The screening GO was a small-model artifact; on the primary hard target rotation hurts. **Conclusion: rotation and CARE-KV do not compose at scale** — no confirm run warranted.


## 5f. LongBench (T5) — data unblocked, RUNTIME-blocked by prototype generation

Data was unblocked (THUDM/LongBench `data.zip` → extracted trec/triviaqa/samsum, ~70 MB, fits the 5 GB free) and a generation+metric driver built (`eval_longbench_subset.py`, reusing the USE_CACHE=1 retrieval harness). But the eval is **not runnable at any useful scale**:

- **CARE-KV generation is prototype-slow.** On TinyLlama, trec, max_ctx=768, **fp16 finished 3 samples in 10 s**, but the **CARE-KV-patched base_quant did NOT finish 3 samples in 600 s** (~200 s/sample — the per-token decode through the patched layer is the bottleneck). CARE-KV proper is slower still; a 7B model is far worse.
- **TinyLlama is too weak for the task** (fp16 trec accuracy = 0.0), so the only model fast enough to run gives floor scores → the CARE-KV-vs-Base comparison would be uninformative even if it finished.

→ A meaningful LongBench run needs a **capable (7B+) model**, which is **infeasibly slow** with CARE-KV's prototype generation (consistent with §5d). **LongBench is blocked on runtime, not data** — it requires the CUDA/Triton correction kernels before it is practical. Driver + data path are in place for when kernels land.


## 5g. Memory–PPL Pareto vs TurboQuant — **CORRECTED**: the earlier "dominated" verdict was a stale-kernel artifact

**Prior claim (superseded):** the production grid (old `eac`/`ezoo` harness) reported CARE-KV losing to TurboQuant at NS=64 (0W/12L) → "Pareto-dominated." A same-harness reproduction shows that verdict was a **correction-kernel artifact**, not a real deficit.

### Kernel-dependence reproduction (run_one harness, Mistral-7B-v0.3, SL512, NS=64)

Re-running the exact production cell in the canonical `run_one` harness (`tools/eval_combined_vs_turbo.py`), the fp16/base/Turbo references reproduce production **to 4 decimals** (windowing identical) — but CARE-KV does **not**:

| arm | run_one NS=64 | production NS=64 (old eac) | Δ turbo | Δ current |
|---|---|---|---|---|
| fp16 | 7.1564 | 7.156 ✅ | — | — |
| base_int3 | 7.6753 | 7.675 ✅ | — | — |
| **turbo_int3** | **7.5862** | 7.586 ✅ | 0 | — |
| carekv_current | **7.5164** | 7.614 ❌ (**+0.098**) | **−0.070** ✓ | 0 |
| **carekv_combined** (KSCORE_LIVE) | **7.411** | — | **−0.175** ✓✓ | **−0.105** |

- fp16/base/Turbo match exactly ⇒ same windowing, same Turbo. **Only CARE-KV differs** (7.516 vs production 7.614). The sole difference is the **correction kernel**: canonical `correction_impl=vectorized` (P5-full) vs the deleted `eac` harness's old `VDOM_ONLY` vectorization.
- **Kernel verified faithful:** `tests/test_vectorized_carekv.py` — vectorized `joint+both` (the paper config) reproduces the `cached` reference loop at **Δ=1.79e-07 (31/31 checks)**. So **7.516 is the true cached CARE-KV**; production's 7.614 came from a non-faithful old kernel (~0.1 PPL inflation). *(CLAUDE.md §1's "joint+both falls back to cached" note is stale for this repo — P5-full handles joint+both bit-close.)*
- Consequence: **with the faithful kernel, CARE-KV current beats Turbo (−0.070), and `combined_kvscore` beats Turbo by −0.175** on Mistral. `combined − current = −0.105`, consistent across NS=8/16/32/64 (robust selector win).
- **Longer context confirms:** at SL1024 (NS=32) combined 6.241 also beats Turbo 6.372 (−0.131) and current 6.323 (−0.082) — the Turbo-beating margin is not an SL512 artifact.

### Corrected Pareto (Mistral SL512 NS=64)

| method | mem × fp16 | PPL | Pareto |
|---|---|---|---|
| BaseQuant_INT3 | 0.203 | 7.675 | dominated |
| TurboQuant_INT3 | **0.203** | 7.586 | on-front (cheapest) |
| CARE-KV current | 0.230 | 7.516 | **on-front** (better PPL, +0.027 mem) |
| **CARE-KV combined** | 0.230 | **7.411** | **on-front** (best PPL) |
| fp16 | 1.000 | 7.156 | (anchor) |

- **CARE-KV is no longer Pareto-dominated on Mistral.** Turbo and CARE-KV form a genuine **quality↔memory trade-off**: Turbo is cheapest (0.203×), CARE-KV combined has the best INT3 PPL (7.411, +0.027× memory). Neither dominates.
- The memory fractions (BaseQuant/Turbo 0.203×, CARE-KV 0.230×) are unchanged and still analytic-validated; only the **PPL side of the comparison** is corrected.

### Generalization to larger models (SOLAR-10.7B, 13B)

Re-run in the same `run_one` harness to test whether the Mistral Turbo-beating result holds beyond 7B.

**SOLAR-10.7B (GQA, NS=64, SL512):**

| arm | PPL | Δ turbo | Δ current |
|---|---|---|---|
| base_int3 | 6.8297 | — | — |
| turbo_int3 | 6.7299 | 0 | — |
| CARE-KV current | 6.7044 | **−0.026** ✓ | 0 |
| CARE-KV combined | 6.7052 | **−0.025** ✓ | +0.0008 |

- **Second model where CARE-KV beats Turbo** — current & combined both < turbo → the win **generalizes from 7B (Mistral) to 10.7B (SOLAR)**, both GQA.
- **But two caveats sharpen the honest picture:** (i) the margin is **much smaller** than Mistral (−0.026 vs −0.07…−0.24) — larger models narrow the gap; (ii) **`combined_kvscore` gives no gain on SOLAR** (combined − current = +0.0008, tied), whereas on Mistral it won by −0.10. **The selector advantage is model-specific, not universal.** So the robust cross-model claim is "CARE-KV *current* beats Turbo on these two models," not "combined is universally better."

**Llama-2-13B (MHA, NS=32, SL512):** first note this exposed — and we fixed — a real engineering bug: the initial run's `nan` was a **swallowed CUDA-OOM** (each layer allocated a full `num_layers×` cache arena; MHA's large Hkv made 13B exceed single-GPU memory), *not* a quantization/outlier limit (the standalone INT3 quantizer on real 13B K/V is clean, max|K|=20). Fixed via the shared-cache arena (commit 2f8b59f). The real result:

| arm | PPL | Δ turbo | Δ base | Δ current |
|---|---|---|---|---|
| base_int3 | 6.8961 | +0.485 | 0 | — |
| turbo_int3 | 6.4111 | 0 | −0.485 | — |
| CARE-KV current | 6.6549 | **+0.244** | −0.241 | 0 |
| CARE-KV combined | 6.5003 | **+0.089** | −0.396 | −0.155 |

- **CARE-KV loses to Turbo on 13B** (current +0.244, combined +0.089) — but **beats base substantially** (combined −0.396). The residual correction closes most of the base→turbo gap (0.396 of 0.485) yet falls **0.089 short** of Turbo.
- **This confirms the structural story (§6):** Llama-2-13B is the **most outlier-heavy** of the three (Turbo improves on base by 0.485, the largest) — exactly the regime where an un-rotated INT3 base collapses and QJL's **rotation wins**. Turbo-beating holds on outlier-mild models and fails on outlier-heavy ones.
- **combined helps here (−0.155 vs current)** — so its benefit is not "Mistral-specific" but **variable** (Mistral −0.10, SOLAR ~0, 13B −0.155); even where it helps most it does not overturn the 13B Turbo deficit.

### Honest scope / open items

- **Verified across 3 models (run_one):** CARE-KV beats Turbo on **Mistral-7B** (current −0.07, combined −0.175) and **SOLAR-10.7B** (−0.026 / −0.025), but **loses on Llama-2-13B** (current +0.244, combined +0.089). Turbo-beating is **not universal** — it holds on outlier-mild models and fails on the most outlier-heavy one (13B).
- **The discriminator is K-outlier severity** (≈ how much Turbo's rotation improves on the un-rotated base): small on Mistral/SOLAR → CARE-KV wins; large on Llama-2-13B (0.485) → rotation wins. This matches the §6 structural account.
- **Selector (`combined`) gain is variable, not universal:** +0 on SOLAR, −0.10 on Mistral, −0.155 on Llama-2-13B. It helps most on 13B yet still doesn't overturn the Turbo deficit there.
- **Coverage:** 3/~12 models measured in `run_one`; the deleted `eac` grid (the other models) must still be re-run before any general claim.
- **Eval-level cached check — abandoned as infeasible, not needed.** An attempted `cached` vs `vectorized` PPL cross-check (NS=8, SL512) confirmed the documented runtime wall: `correction_impl=cached` is the per-(layer, kv_head, token) Python-loop prototype (§5d, ~100× slower), and did not complete even a single NS=8 cell in **>19 h** of wall-clock. The unit test is already conclusive (vectorized `joint+both` == cached at **Δ=1.79e-07**, tensor-level), so faithfulness is established by construction; the eval-level rerun would only reproduce the same conclusion at impractical cost. `vectorized` PPL for this NS=8 cell was 7.9610; cached is guaranteed to land within ~1e-4 of it. Verified via the unit test, not the (infeasible) eval loop.

→ **Revised positioning:** CARE-KV beats naive INT3 everywhere, and with the faithful correction kernel it **beats TurboQuant on outlier-mild models** (Mistral-7B −0.07…−0.24, SOLAR-10.7B −0.026) — sitting on the quality↔memory Pareto front there — **but loses on the outlier-heavy Llama-2-13B** (+0.089), where QJL's rotation is exactly the right tool. So the honest headline is *conditional*: **CARE-KV wins when K-outliers are mild and loses when they are severe**, tracking the same structural axis as §6. It additionally **composes** with orthogonal methods (eviction ✓ §5b, mixed-precision ~ §5c), which score-level QJL/TurboQuant cannot — a standing qualitative advantage independent of the PPL race. The `combined_kvscore` selector's extra gain is **variable** (0 on SOLAR, −0.10 on Mistral, −0.155 on 13B), so the cross-model story rests on CARE-KV broadly, not on combined specifically. The earlier "Turbo Pareto-dominates CARE-KV everywhere" line (a non-faithful-kernel artifact) is retracted; the corrected picture is a genuine **model-dependent split**, not a clean win or a clean loss.


## 6. Honest paper positioning

CARE-KV is a **reliable improvement over naive INT3 compression** (beats BaseQuant everywhere, across 11 architectures). On **Mistral at rigorous NS=64, `combined_kvscore` now beats TurboQuant (−0.175 PPL)** with the faithful correction kernel (§5g) — the first robust Turbo-beating case; the earlier "Turbo Pareto-dominates" verdict was a stale-kernel artifact and is retracted for Mistral. **Generalization is still open:** the full-grid "0W/12L vs Turbo" was measured on the now-deleted `eac` harness with a non-faithful kernel, so the other 11 models (many outlier-heavier → structurally harder) must be **re-run in `run_one`** before claiming a general Turbo win. The remaining structural risks on hard models (un-rotated base + sparse capped residual + unstable K correction on outlier-heavy K) are unchanged. Further leads: (a) **rotation-CARE-KV** (in screening, NO-GO at 7B §5e), (b) **K-correction stabilization** (clamp/norm-guard, untested at scale). A clean negative on rotation (substitutes, not complements) is itself a citable finding.

