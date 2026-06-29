# K-score ablation (Phase 3) — SMOKE

> **Smoke ablation, NOT the final full grid.** Model Mistral-7B-v0.3, seq_len [128, 256], num_samples 4. Flag-gated query-aware K-side residual score (`CAREKV_KSCORE_LIVE`); with `CAREKV_KSCORE_LIVE=0` the path is byte-identical to the current V-score-only CARE-KV. Variants: vscore_only / kscore_only / combined_kvscore (λ_k=1.0, λ_v=1.0). Failures/OOMs preserved.

## PPL by variant and seq_len

| SL | fp16 | base INT3 | vscore_only | kscore_only | combined_kvscore | best | K-score changed decision | combined beats vscore |
|---|---|---|---|---|---|---|---|---|
| 128 | 8.6728 | 9.0214 | 8.8176 | 9.0214 | 8.8485 | **vscore_only** | no | no |
| 256 | 7.356 | 7.7264 | 7.6007 | 7.7264 | 7.6785 | **vscore_only** | no | no |

## Effective budgets
- vscore_only → SK0SV4 (V-dominant)
- kscore_only → SK2SV0 (K-dominant)
- combined_kvscore → SK2SV4 (K+V)

## Failures / OOM (preserved): 0


## Notes
- This is a SMOKE ablation; do not read it as the final full K-score grid.
- `CAREKV_KSCORE_LIVE=0` (vscore_only) is the unchanged default V-score-only path.

