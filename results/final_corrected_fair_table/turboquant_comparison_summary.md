# CARE-KV vs TurboQuant — fair INT3 (audited)

WikiText-2 PPL (lower=better), fixed INT3 bit-width. Verdict = audited `fair_int3_result`. Source: `final_quality_main_table.csv`.

| Model | Seq | fp16 | BaseQuant | **CARE-KV** | TurboQuant | ΔCARE−Turbo | ΔCARE−Base | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| Mistral-7B-v0.3 | 256 | 7.355 | 7.801 | **7.693** | 7.686 | +0.008 | -0.107 | ≈ tie |
| Mistral-7B-v0.3 | 512 | 6.785 | 7.297 | **7.125** | 7.489 | -0.364 | -0.172 | ✅ CARE-KV win |
| Mistral-7B-v0.3 | 1024 | 6.280 | 6.715 | **6.618** | 6.608 | +0.010 | -0.097 | ≈ tie |
| Yi-6B | 256 | 8.652 | 8.749 | **8.749** | 8.774 | -0.025 | +0.000 | ≈ tie |
| Yi-6B | 512 | 7.960 | 8.483 | **8.473** | 8.255 | +0.218 | -0.010 | ✗ TurboQuant win |
| Yi-6B | 1024 | 6.947 | 7.697 | **7.502** | 7.382 | +0.119 | -0.196 | ✗ TurboQuant win |
| deepseek-llm-7b-base | 256 | 10.160 | 11.356 | **11.206** | 10.768 | +0.438 | -0.151 | ✗ TurboQuant win |
| deepseek-llm-7b-base | 512 | 8.598 | 9.607 | **9.460** | 9.058 | +0.402 | -0.148 | ✗ TurboQuant win |
| deepseek-llm-7b-base | 1024 | 7.822 | 8.889 | **8.889** | 8.283 | +0.607 | +0.000 | ✗ TurboQuant win |
| open_llama_7b_v2 | 256 | 9.084 | 9.498 | **9.418** | 9.528 | -0.111 | -0.081 | ✅ CARE-KV win |
| open_llama_7b_v2 | 512 | 8.393 | 8.889 | **8.870** | 8.748 | +0.122 | -0.019 | ✗ TurboQuant win |
| open_llama_7b_v2 | 1024 | 7.650 | 8.164 | **8.097** | 8.040 | +0.057 | -0.068 | ✗ TurboQuant win |

**Tally — CARE-KV vs TurboQuant INT3: 2 win / 3 tie / 7 loss.** (vs BaseQuant INT3: CARE-KV never worse.)

## Settings where CARE-KV BEATS TurboQuant

| Model | Seq | CARE-KV | TurboQuant | Δ |
|---|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 512 | **7.125** | 7.489 | **-0.364** |
| open_llama_7b_v2 | 256 | **9.418** | 9.528 | **-0.111** |

**Reading.** CARE-KV's clearest win is **Mistral-7B SL512 (7.125 vs 7.489, −0.36)**. It is *competitive, not uniformly superior*: TurboQuant wins the diffuse-error settings (DeepSeek-7B, long-context Yi/OpenLLaMA) where QJL's score-level correction — which CARE-KV cannot stack — dominates. CARE-KV's robust advantage is over the same-bit BaseQuant INT3 baseline (never worse).

**Status: audited fair-INT3 (N=4).**
