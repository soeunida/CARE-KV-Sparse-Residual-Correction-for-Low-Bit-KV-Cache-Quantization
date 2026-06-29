# Low-rank dense correction — design (direction ③)

> Follows the rotation+CARE-KV NO-GO and the coverage ablation (finer-
> granularity rejected). Goal: close the diffuse-error gap to TurboQuant
> (DeepSeek-7B, long-context Yi/OpenLLaMA) without rotation and without QJL
> (unstackable). **Phase-0 result: NO-GO — the un-rotated INT3 residual is not
> low-rank (see RECOVERED_rotation_lowrank_results.md §5).** Kept for the record.

## 0. Core insight
- CARE-KV's sparse residual works when INT3 error is *concentrated*; fails when
  *diffuse*. Rotation makes error full-rank/uniform → sparse can't select it.
- The **un-rotated** INT3 residual is dominated by a few outlier channels across
  ~all tokens → low effective rank in the channel subspace (HYPOTHESIS).
- → model that structure with a small dense low-rank correction.

## 1. Method
Per (layer, kv_head), X ∈ {K_post, V}, base reconstruction X̂:
```
R = X − X̂                      # (T, D)
P = top-r right singular vectors of R   # (D, r) channel basis, shared over tokens
C = R @ P                       # (T, r) per-token coefficients
X̂' = X̂ + C @ Pᵀ
```
P computed once per sequence at prefill (thin/randomized SVD); C is the only
per-token cache state added (r numbers/token). Reconstruction is a rank-r matmul.

## 2. Stack
INT3 base + low-rank dense (global diffuse directions) + (optional) sparse
CARE-KV (local residual of the low-rank fit). Complementary; no rotation.

## 3. Memory
Per token: r coeffs × (2B fp16 | 1B int8). Amortized P = D·r·2 B per (layer,head).
TinyLlama D=64, r=2, fp16 ≈ +17%/kind vs INT3 base. Must report vs sparse at
equal memory.

## 4. Hypotheses
- H1 low-rank beats sparse on diffuse settings at equal memory.
- H2 small r (1–2) captures most gain (verify via SVD spectrum).
- H3 low-rank + sparse > either alone.
- H4 no rotation needed (rotation destroys the low-rank structure).

## 5. Phased plan
- Phase 0 (eval-mode, done): X̂'=X̂+R P Pᵀ in prefill only (carekv_eval-style),
  base+low-rank, ranks 1/2/4/8 → screen quality. **Result: NO-GO** (rank-8 upper
  bound 14.02 < sparse bar 13.46 is FALSE; 14.02 > 13.46, worse + costlier).
- Phase 1 SVD-energy spectrum (confirm low rank / pick r). Not needed given NO-GO.
- Phase 2 cache+decode integration (only if Phase 0 won). Not pursued.
- Phase 3 4-model confirm. Not pursued.

## 6. Implementation surface
- Phase 0: `_lowrank_correct_eval` in layer.py (gated to base_quant via
  CAREKV_LOWRANK_RANK), `tools/eval_lowrank_dense.py`. (recovered)

## 7. Decision
NO-GO: residual not low-rank enough; sparse CARE-KV already near best value-level
correction for the concentrated un-rotated residual. The diffuse-setting gap to
TurboQuant is fundamentally a score-level (QJL) phenomenon not stackable here.
