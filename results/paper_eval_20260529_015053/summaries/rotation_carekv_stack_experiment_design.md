# Rotation + CARE-KV stack — experiment design

> Builds on the negative post-RoPE Hadamard pilot and the TurboQuant comparison.
> Goal: can a rotation-improved base + CARE-KV residual close the diffuse-error
> gap to TurboQuant, and under which configuration? **Outcome: NO-GO at RV=2;
> WEAK-GO only via rotation + read-breadth (see
> RECOVERED_rotation_lowrank_results.md §2,§4).**

## 0. Why
TurboQuant beats CARE-KV in diffuse settings via (a) random rotation (spreads
outliers full-rank) + (b) QJL unbiased score correction. CARE-KV has neither.
Idea: prepend rotation to CARE-KV's *base* (value-level, composes), keep the
sparse residual. **Hard ceiling:** QJL is score-level, method-incompatible with
CARE-KV's value-level correction → a stack can only inherit rotation, never QJL.

## 1. Prior bar
uniform INT3 16.20; uniform+CARE-KV **13.46** (bar); Hadamard post-RoPE
standalone 27.45; Hadamard post-RoPE + CARE-KV 15.23 (naive stack failed —
rotation applied post-RoPE, where K is already RoPE-mixed).

## 2. Hypotheses
- H1 pre-RoPE rotation ≫ post-RoPE; pre-RoPE + CARE-KV beats 13.46.
- H2 random-Gaussian vs Walsh-Hadamard rotation.
- H3 complementarity: rotation makes error uniform → does sparse top-k help less
  (substitute) or does smaller residual help per slot (complement)?
- H4 gain largest where CARE-KV loses to TurboQuant (DeepSeek, long ctx).

## 3. Arms (INT3, value-level)
uniform / uniform+CARE-KV(bar) / Hadamard post-RoPE+CARE / Hadamard pre-RoPE+CARE
/ random pre-RoPE+CARE / random pre-RoPE standalone / TurboQuant full (ref).

## 4. Protocol
WikiText-2 PPL via baselines KVMethodAdapter harness; equal-memory; report
reads/runtime; invariant rotation R=I reduces to per-channel quant; screen
TinyLlama N=4→N=16, gate before 7B.

## 5. Implementation surface
- randrot base quantizer + injectable rotate-quant core (kivi_helpers.py).
- pre-RoPE store mode for rotation quantizers (layer.py prefill+decode), reuse
  kvquant pre_rope re-apply-RoPE machinery.
- side-buffer registration (cache.py), adapter pre/post tags + randrot
  (baselines/carekv_adapter.py).
- invariant test (tests/test_rotation_base_invariant.py).

## 6. Result summary
- H1 confirmed (post 15.23 → pre 17.9-ish reaches parity).
- N=16 RV=2: best stack ties bar → NO-GO.
- N=16 read-breadth (RV4) on rotated base beats bar by −0.33 → WEAK-GO.
- random < Hadamard at scale. Standalone rotated base ≤ uniform base.
- Next: 7B diffuse confirm (DeepSeek, long-ctx), beware 7B non-transfer.
