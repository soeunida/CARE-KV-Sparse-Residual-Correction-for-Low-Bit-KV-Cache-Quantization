# CARE-KV long-context evaluation (SL ≥ 4096)

Model: **deepseek-ai/deepseek-llm-7b-base**. Regime where the KV cache is the actual bottleneck. Labels follow CLAUDE.md §9.3: **[real]** = measured forward, **[analytical]** = estimator, **[blocked]** = prototype runtime/memory limited.

## 1. PPL vs sequence length  [real]

fp16 and BaseQuant INT4/INT3 KV (KIVI-style per-channel-K / per-token-V fake-quant hook — model-agnostic, not the CARE-KV Python-loop prefill, so it runs at SL ≥ 4096).

| SL | fp16 PPL | BaseQuant INT4 | BaseQuant INT3 | fp16 tok/s | peak GPU (MB) |
|---:|---:|---:|---:|---:|---:|
| 2048 | 6.2679 | 6.4044 | 10.3970 | 5625.3 | 16363 |
| 4096 | 6.4145 | 6.5886 | 12.6210 | 6110.1 | 18896 |

## 2. KV-cache memory vs sequence length  [analytical]

Per-sequence KV memory (GB) from the repository estimator. This is the direct evidence that the KV cache dominates at long context, and that CARE-KV (INT3 base + sparse residual) keeps it small.

| SL | fp16 KV (GB) | BaseQuant INT3 (GB) | CARE-KV INT3 total (GB) | CARE-KV / fp16 |
|---:|---:|---:|---:|---:|
| 2048 | 0.9375 | 0.1758 | 0.2159 | 0.230× |
| 4096 | 1.8750 | 0.3516 | 0.4319 | 0.230× |

## 3. CARE-KV at long context — status  [blocked / projected]

CARE-KV's **PPL** at SL ≥ 4096 is **not measured here**: the paper method (`carekv_stored`, `correction_impl=cached`) uses a per-(layer, kv_head, token) **Python-loop prefill**, which is multi-hour at long context, and the HF `DynamicCache` dummy-fp16 K/V inflates peak GPU memory past a 49 GB card at SL=4096.

Measured prototype cost at SL=1024 (DeepSeek-7B, N=4), for scale:

| SL | mode | runtime (s) | peak GPU (MB) |
|---:|---|---:|---:|
| 512 | carekv_stored INT3 | 922 | 26324 |
| 1024 | base_quant INT3 | 1143 | 41844 |
| 1024 | carekv_stored INT3 | 2921 | 33896 |

Extrapolated to SL=4096 (≈ super-linear in SL) these cells are **multi-hour and likely OOM > 49 GB** — the documented "runtime-blocked by prototype gen" (CLAUDE.md §8). CARE-KV's **KV memory** at long context is reported analytically in §2 (unaffected by the runtime blocker); its **quality** is anchored to the short-context sweep (see the `memory-projected` rows' `note` column). This unblocks once the vectorized joint+both correction / lightweight HF cache land (CLAUDE.md §8).

