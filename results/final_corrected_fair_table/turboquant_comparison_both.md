# CARE-KV vs TurboQuant — fair INT3 (two verdict views)

WikiText-2 PPL, fixed INT3. Same audited PPLs, two classifications.
Source: `final_quality_main_table.csv`.

- **View 1 (margin ±0.02, 11 active settings):** **3 win / 2 tie / 6 loss** (1 n/a).
- **View 2 (audited `fair_int3_result`, 12 settings):** **2 win / 3 tie / 7 loss**.

| Model | Seq | BaseQuant | CARE-KV | TurboQuant | ΔCARE−Turbo | View1 (±0.02) | View2 (audited) |
|---|---:|---:|---:|---:|---:|:--:|:--:|
| Mistral-7B-v0.3 | 256 | 7.801 | **7.693** | 7.686 | +0.008 | ≈ tie | ≈ tie |
| Mistral-7B-v0.3 | 512 | 7.297 | **7.125** | 7.489 | -0.364 | ✅ win | ✅ win |
| Mistral-7B-v0.3 | 1024 | 6.715 | **6.618** | 6.608 | +0.010 | ≈ tie | ≈ tie |
| Yi-6B | 256 | 8.749 | **8.749** | 8.774 | -0.025 | ✅ win | ≈ tie |
| Yi-6B | 512 | 8.483 | **8.473** | 8.255 | +0.218 | ✗ loss | ✗ loss |
| Yi-6B | 1024 | 7.697 | **7.502** | 7.382 | +0.119 | ✗ loss | ✗ loss |
| deepseek-llm-7b-base | 256 | 11.356 | **11.206** | 10.768 | +0.438 | ✗ loss | ✗ loss |
| deepseek-llm-7b-base | 512 | 9.607 | **9.460** | 9.058 | +0.402 | ✗ loss | ✗ loss |
| deepseek-llm-7b-base | 1024 | 8.889 | **8.889** | 8.283 | +0.607 | — n/a | ✗ loss |
| open_llama_7b_v2 | 256 | 9.498 | **9.418** | 9.528 | -0.111 | ✅ win | ✅ win |
| open_llama_7b_v2 | 512 | 8.889 | **8.870** | 8.748 | +0.122 | ✗ loss | ✗ loss |
| open_llama_7b_v2 | 1024 | 8.164 | **8.097** | 8.040 | +0.057 | ✗ loss | ✗ loss |

## View 1 (margin ±0.02) — CARE-KV wins (3)

| Model | Seq | CARE-KV | TurboQuant | Δ |
|---|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 512 | **7.125** | 7.489 | **-0.364** |
| open_llama_7b_v2 | 256 | **9.418** | 9.528 | **-0.111** |
| Yi-6B | 256 | **8.749** | 8.774 | **-0.025** |

## View 2 (audited) — CARE-KV wins (2)

| Model | Seq | CARE-KV | TurboQuant | Δ |
|---|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 512 | **7.125** | 7.489 | **-0.364** |
| open_llama_7b_v2 | 256 | **9.418** | 9.528 | **-0.111** |

**Difference between the two views:** identical PPLs; View 2 reclassifies **Yi-6B SL256** (−0.025) as a tie and includes **DeepSeek SL1024** (CARE-KV fell back to BaseQuant) as a loss. Clearest win either way: **Mistral-7B SL512 (−0.364)**. CARE-KV is competitive with TurboQuant, not uniformly superior; its robust advantage is over same-bit BaseQuant INT3 (never worse).
