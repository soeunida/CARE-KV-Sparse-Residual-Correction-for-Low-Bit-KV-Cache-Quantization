# CARE-KV Memory Optimization Audit (Phase A)

Reference config (TinyLlama-like) used for all numbers below:

```
num_layers     = 22
num_heads      = 32     (query heads)
num_kv_heads   = 4      (GQA 8x)
head_dim       = 64
page_size      = 16
max_pages      = 128    → 2,048-token capacity per (layer, KV head)
base_bits      = 3      (INT3)
group_size     = 32
k_channel_group= 32
v_token_block  = 4
store_budget   = 0.10
sketch_dim     = 16
```

FP16 reference at capacity: **46.14 MB** (`L · Hkv · T · D · 2 · 2` bytes).

## 1. Buffers actually allocated at runtime

From `cache.CAREKVCache._init_buffers` (cache.py:114-147):

| Buffer | Shape | Dtype | Actual MB |
|---|---|---|---:|
| `base_K_codes` | `(L, Hkv, P, T, D)` | `int8` | **11.53** |
| `base_V_codes` | `(L, Hkv, P, T, D)` | `int8` | **11.53** |
| `base_K_scale` | `(L, Hkv, P, T, G)` | `fp16` | 0.72 |
| `base_V_scale` | `(L, Hkv, P, T, G)` | `fp16` | 0.72 |
| `valid_tokens` | `(L, Hkv, P)` | `int32` | 0.05 |
| `k_residual_buf` | `(max_slots, k_slot_size)` | `int8` (4-bit packed) | 1.15 |
| `v_residual_buf` | `(max_slots, v_slot_size)` | `int8` (4-bit packed) | 0.58 |
| `k_residual_scale` | `(max_slots, 1)` | `fp16` | 0.01 |
| `v_residual_scale` | `(max_slots, 1)` | `fp16` | 0.01 |
| metadata (extrapolated from one PageMeta × L·Hkv·P) | — | mixed | **2.12** |
| **Actual total at 2048-token capacity** | | | **28.42** MB |

`max_slots` for the residual buffers is sized at allocation time by `cache.py:131`:
`max_slots = max(64, int(P · Hkv · L · store_budget_ratio · 4))` → here 4,505.

> Note: `k_residual_buf` and `v_residual_buf` are sized for the entire cache
> (sum across layers/heads/pages), not per-page, so allocation is a fixed
> arena. Pages reserve slots via `_k_slot_free` / `_v_slot_free`.

## 2. Buffers included in `estimate_memory_bytes`

From `utils.estimate_memory_bytes` (utils.py:28-130). Returns 14 keys:

| Estimator key | What it counts | Backed by? |
|---|---|---|
| `base_K_code_bytes` / `base_V_code_bytes` | Theoretical packed bytes for `L·Hkv·tokens_padded·D` at `base_bits` | ❌ runtime is int8 |
| `base_K_scale_bytes` / `base_V_scale_bytes` | `L·Hkv·tokens_padded·G·2` (fp16) | ✅ matches |
| `residual_K_bytes` | `int(total_k_cands · store_ratio) · (packed4_size + 2)` | ✅ matches |
| `residual_V_bytes` | `int(total_v_cands · store_ratio) · (packed4_size + 2)` | ✅ matches |
| `metadata_bytes` | per-page indices: k_slots(int32) + v_slots(int32) + valid_tokens + token_start + page_id | partially — the runtime uses Python lists (~8 B/entry) not int32, plus dataclass overhead |
| `error_norm_bytes` | `(num_k_cg + num_v_blk) · 2 · num_pages · Hkv · L` (fp16) | ✅ matches |
| `sketch_bytes` | `num_k_cg · sketch_dim · 2 · num_pages · Hkv · L` (fp16) | ✅ matches |
| `total_bytes` | sum | derived |
| `fp16_kv_bytes` / `int4_kv_bytes` | references using `Hkv` (GQA-correct) | derived |
| `compression_vs_fp16` / `compression_vs_int4` | ratios | derived |
| `packed_mode` | the flag the caller passed | derived |

## 3. Base KV codes: int8 or packed?

**int8 in storage today**, regardless of `base_bits`.

The `int8` representation in `base_K_codes` / `base_V_codes` holds *values in the
range* of the chosen bit-width (e.g. `[-4, 3]` for INT3) but each value is
still one byte. So an INT3 cache uses 2.67× the storage of a tight-packed
INT3 cache.

`CacheConfig.packed_storage` exists in code (cache.py:73) but **no codepath
honours it** — the buffer dtype is hard-wired to `torch.int8`. This is the
single biggest gap between actual allocation and the estimator's `packed=True`
mode (which is what the 0.271× headline reports).

## 4. INT3 packing

**Estimator-only at present.**

- INT2 packing exists in `quantizer.py:pack_int2`/`unpack_int2` — round-trip-tested.
- INT4 packing exists in `quantizer.py:pack_int4`/`unpack_int4` — round-trip-tested.
- INT3 packing does **not** exist. The estimator computes
  `_bits_to_packed_bytes(D, 3) = ceil(D·3/8)` bytes per token but no runtime
  packer is implemented.

The user-stated goal "27.1% of FP16" relies on INT3 packing being real, which it currently is not. Phase B should implement it; the natural format is 8 signed-3-bit codes (24 bits) per 3 bytes.

## 5. Scales

**fp16 storage**, contributing **11.52%** of the packed-mode total.

- `base_K_scale` / `base_V_scale`: `(L, Hkv, T, G)` fp16, **0.72 MB each** (1.44 MB combined).
- `k_residual_scale` / `v_residual_scale`: one fp16 scale per packed slot, negligible (<0.02 MB total).
- Per-page sketches and error norms are also fp16 (Phase A counts them under
  `sketch_bytes` / `error_norm_bytes`).

At INT3 with packed base codes the scales become 11.5% of the cache — large enough that **Phase C (bf16 / int8 scales)** is worth implementing.

## 6. Residual slots

**9.32%** of total (packed). At store_budget=0.10:
- K slots: `int(num_k_cands · 0.10) · (k_slot_packed4 + 2 B scale)`
- V slots: similar with v_token_block-sized payload
- Each slot is one int8 row in `k_residual_buf` / `v_residual_buf` (4-bit packed values).

Slot allocation is from a free-list; the buffer is sized for the worst case
(`store_budget_ratio · 4` over-allocation factor in cache.py:131), so the
actual arena is larger than the "used" portion. **Estimator reports only the
used portion; actual reserved arena is up to 4× that for worst-case
fragmentation.** Not a paper concern, but worth noting.

## 7. Metadata

**3.24%** of packed total (estimator) — `0.41 MB` for 11,264 pages at this
config. Per-page: ~36 bytes (two small int32 slot-id lists + valid_tokens +
token_start + page_id).

Runtime metadata is larger than the estimator says — Python lists carry per-
element overhead (~8 B vs the estimator's 4 B int32 assumption) and PageMeta
is a dataclass instance (~64 B object overhead). The extrapolated runtime
total is **2.12 MB** instead of 0.41 MB, ~5× higher. For paper memory tables
this is fine to keep at the estimator-side packed number, since a real
production implementation would replace the dataclass with a struct-of-arrays.
**Flag as a known reporting gap.**

## 8. Sketches

**5.76%** of packed total — `0.72 MB`. Per page: `num_k_cg · sketch_dim · 2 B`
= `2 · 16 · 2` = 64 B in this config. Total = 11,264 pages × 64 B.

Halving `sketch_dim` from 16 → 8 saves 0.36 MB (~3% of total). Phase D should
sweep this.

## 9. GQA: KV heads or query heads?

**KV heads.** After the Phase 2 rewrite, all of these index by `Hkv`:
- `base_K/V_codes` shape uses `Hkv` (cache.py:118).
- `valid_tokens` uses `Hkv` (cache.py:124).
- `meta_table[L][Hkv][P]` (cache.py:140).
- `next_page[L][Hkv]` (cache.py:146).
- `fp16_kv_bytes` reference in the estimator uses `Hk` (utils.py:121).
- `residual_K_bytes` candidate count uses `Hk` (utils.py:84).

For TinyLlama (Hq=32, Hkv=4) this is an 8× saving over the pre-Phase-2 layout.

## 10. Mismatches between actual allocation and estimator

| Item | Estimator (packed) | Actual | Ratio |
|---|---:|---:|---:|
| Base K codes | 4.33 MB | 11.53 MB | **2.67× (INT3-as-int8)** |
| Base V codes | 4.33 MB | 11.53 MB | **2.67×** |
| Scales | 1.44 MB | 1.44 MB | 1.00× ✓ |
| Residual slots | 1.17 MB | 1.73 MB allocated (arena), ≤ 1.17 used | up to 1.48× arena over-allocation |
| Metadata | 0.41 MB | 2.12 MB | **5.16× (Python overhead)** |
| Sketches | 0.72 MB | 0.72 MB | 1.00× ✓ |
| Error norms | 0.14 MB | 0.14 MB | 1.00× ✓ |
| **Total (capacity)** | **12.52 MB** | **28.42 MB** | **2.27×** |
| **vs FP16** | **0.271×** | **0.616×** | — |

The headline "0.271× FP16" is **achievable but not yet real** — it requires:
1. Phase B: INT3 packed base storage (closes 14.4 MB / 2.67× gap on codes).
2. Cosmetic: replace per-page PageMeta dataclass with arrays (closes 1.7 MB).

After Phase B + metadata cleanup, actual ≈ 12.3 MB ≈ 0.27× FP16. **Until then, the estimator's packed number is upper-bound theoretical and the int8 number (0.616×) is the true on-device cost.**

## Concrete recommendations for downstream phases

- **Phase B**: implement uint3 packing (8 codes per 3 bytes), wire through
  `CacheConfig.packed_base`, decode-on-the-fly in `read_base_concat`. Highest
  ROI by far.
- **Phase C**: bf16 scales are zero-cost relative to fp16 (same width).
  Real saving requires int8 scales, but PPL impact must be measured before
  enabling.
- **Phase D**: largest residual-side wins probably come from
  - `sketch_dim` 16 → 8 (saves 0.36 MB)
  - `v_token_block` 4 → 8 (halves V slot count)
  - `k_channel_group` 32 → 64 (halves K slot count and sketch count)
  PPL-vs-memory Pareto should drive choice.
- **Phase E**: layer_sensitivity budgets can let `store_budget` average remain at 0.10
  while concentrating slots in high-sensitivity layers.
- Replace `meta_table` Python lists with int32 `slot_index` tensors of shape
  `(L, Hkv, P, num_k_cg)` and `(L, Hkv, P, num_v_blk)` for honest metadata
  accounting. Estimator-only fix is also acceptable if labelled.
