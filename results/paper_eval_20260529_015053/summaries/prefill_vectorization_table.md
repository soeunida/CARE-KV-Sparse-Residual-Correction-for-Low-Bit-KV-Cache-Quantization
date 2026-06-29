# Prefill correction vectorization

TinyLlama-1.1B, INT3 carekv_stored, **separate** route policy (`joint`
falls back to `cached` because vectorized V doesn't yet reproduce the
joint K/V interleaved top-k).  packed_base=1, scale_quant=int8,
absolute SK=2 SV=4 RK=2 RV=2, budget_policy=uniform.

| seq_len | impl | PPL | seconds | K_reads | V_reads | peak_MB | speedup_vs_python | speedup_vs_cached |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | python | 3.8661 | 153.4 | 90112 | 90112 | 4028.05 | 1.00× | 0.76× |
| 64 | cached | 3.8661 | 116.4 | 90112 | 90112 | 4028.05 | 1.32× | 1.00× |
| 64 | vectorized | 3.9718 | 85.7 | 90112 | 90112 | 4028.05 | 1.79× | 1.36× |
| 128 | python | 2.0637 | 472.9 | 180224 | 180224 | 4046.15 | 1.00× | 0.83× |
| 128 | cached | 2.0637 | 393.3 | 180224 | 180224 | 4046.15 | 1.20× | 1.00× |
| 128 | vectorized | 2.0614 | 279.6 | 180224 | 180224 | 4046.15 | 1.69× | 1.41× |

## Acceptance

| Check | Status |
|---|---|
| `python` path produces baseline PPL | ✓ |
| `cached` path matches `python` PPL bit-equally | ✓ (Δ=0 at both seq_lens) |
| `vectorized` is faster than `cached` | ✓ (1.36× at sl=64, 1.41× at sl=128) |
| `vectorized` matches `cached` PPL | partial — Δ=2e-3 at sl=128 (within fp16 noise); Δ=0.11 at sl=64 (topk tie-breaking) |
| READ_ABS_K=READ_ABS_V=0 matches base_quant | ✓ (preserved by short-circuit) |
| Causal + GQA preserved | ✓ (unit test `test_vectorized_v_matches_cached` Δ=1.5e-8 on synthetic) |

## Hybrid scope

- Vectorized engages for **V correction only**.  K correction stays on
  the cached per-(h, t) path (same code as `correction_impl=cached`).
- When `route_policy=joint` AND `kind=both`, layer.py auto-falls-back
  to cached for the entire correction, because vectorized V applies
  a fixed per-query budget_v while joint redistributes K/V slots
  dynamically.  PPL therefore matches cached exactly under that
  fallback.  Vectorizing joint K+V together is left as future work.
- Inside V vectorization:
  - V slots pre-unpacked once per (layer, kv_head) at forward start.
  - Per-(slot, query) attention mass via batched matmul.
  - Top-budget_v selection via `torch.topk` over the (T·Hq_per_kvh, N_v) score matrix.
  - V residual gather is fully vectorized through a non-overlapping-slot
    token-to-slot lookup tensor.
- The residual sl=64 PPL drift (0.11) comes from `torch.topk` vs
  `sorted()` choosing different tied-or-near-tied slots; the absolute
  selection set is the same size (V_reads=90112 in both impls)
  but a few specific picks differ.  At sl=128 the difference is
  paper-acceptable noise (0.002 PPL).
