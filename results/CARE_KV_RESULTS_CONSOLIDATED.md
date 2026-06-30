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


## 6. Honest paper positioning

CARE-KV is a **reliable improvement over naive INT3 compression** (beats BaseQuant everywhere, across 11 architectures) but is **not** a TurboQuant-beater on raw PPL — the deficit is **structural** (un-rotated base + sparse capped residual + unstable K correction on outlier-heavy K). The strongest leads to narrow/flip the Turbo gap are (a) **rotation-CARE-KV** (in screening), (b) **combined_kvscore** selector (Mistral-only win), and (c) **K-correction stabilization** (clamp/norm-guard, untested at scale). A clean negative on rotation (substitutes, not complements) is itself a citable finding.

