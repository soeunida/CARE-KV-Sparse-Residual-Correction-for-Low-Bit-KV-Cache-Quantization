# FINAL quality fair comparison (INT3-only, bit-width-fair)

> Fair comparison is **INT3-only** among BaseQuant_INT3 / Adaptive_CAREKV_INT3 / TurboQuant_INT3_standalone; **tie ≤ 0.02 PPL**. INT4 is higher-bit reference only (not in this table). INT2 is excluded from the main table (recorded `unstable_outlier_collapse`, `paper_usable=no` in the failure table). TurboQuant+CARE-KV stays `unsupported`. Source: score-aware chunked deterministic run (chunked cs=128, TF32-off, attention_output V-score). All OOM/failed/unsupported rows preserved.

## Main quality table (fair INT3)

| model | SL | fp16 | BaseQ INT3 | CARE-KV INT3 (sel / budget) | TurboQ INT3 | **fair INT3 result** | tied methods | notes |
|---|---|---|---|---|---|---|---|---|
| Mistral-7B-v0.3 | 256 | 7.3547 | 7.8006 | 7.6933 (Vdom / SK0SV4) | 7.6855 | **tie** | TurboQuant_INT3_standalone and Adaptive_CAREKV_INT3 |  |
| Mistral-7B-v0.3 | 512 | 6.7854 | 7.2972 | 7.1249 (KV / SK2SV4) | 7.4885 | **Adaptive_CAREKV_INT3** |  |  |
| Mistral-7B-v0.3 | 1024 | 6.2797 | 6.7147 | 6.6181 (KV / SK2SV4) | 6.608 | **tie** | TurboQuant_INT3_standalone and Adaptive_CAREKV_INT3 |  |
| Yi-6B | 256 | 8.6524 | 8.7492 | 8.7492 (BaseQuant_INT3 / SK0SV0) | 8.7745 | **tie** | Adaptive_CAREKV_INT3 and BaseQuant_INT3 | base near-lossless; CARE-KV skipped (no over-claim) |
| Yi-6B | 512 | 7.9602 | 8.4828 | 8.473 (KV / SK2SV4) | 8.2555 | **TurboQuant_INT3_standalone** |  |  |
| Yi-6B | 1024 | 6.9466 | 7.6975 | 7.5016 (Vdom / SK0SV4) | 7.3822 | **TurboQuant_INT3_standalone** |  |  |
| deepseek-llm-7b-base | 256 | 10.1599 | 11.3562 | 11.2056 (Vdom / SK0SV4) | 10.7681 | **TurboQuant_INT3_standalone** |  |  |
| deepseek-llm-7b-base | 512 | 8.5982 | 9.6075 | 9.4598 (KV / SK2SV4) | 9.058 | **TurboQuant_INT3_standalone** |  |  |
| deepseek-llm-7b-base | 1024 | 7.822 | 8.8892 | 8.8892 (BaseQuant_INT3 / no_valid_carekv) | 8.2825 | **TurboQuant_INT3_standalone** |  | CARE-KV Gate B FAIL → fell back to base |
| open_llama_7b_v2 | 256 | 9.0836 | 9.4984 | 9.4176 (Vdom / SK0SV4) | 9.5285 | **Adaptive_CAREKV_INT3** |  |  |
| open_llama_7b_v2 | 512 | 8.3935 | 8.8893 | 8.8699 (KV / SK2SV4) | 8.7476 | **TurboQuant_INT3_standalone** |  |  |
| open_llama_7b_v2 | 1024 | 7.6504 | 8.1643 | 8.0965 (KV / SK2SV4) | 8.0399 | **TurboQuant_INT3_standalone** |  |  |

**Fair INT3 tally:** TurboQuant_INT3_standalone=7, tie=3, Adaptive_CAREKV_INT3=2.

## Failure / collapse / unsupported (preserved)

| model | SL | method | ppl | status | paper_usable | reason |
|---|---|---|---|---|---|---|
| Mistral-7B-v0.3 | 256 | TurboQuant_INT2_standalone | 48.7442 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| Mistral-7B-v0.3 | 256 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| Mistral-7B-v0.3 | 512 | TurboQuant_INT2_standalone | 61.2083 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| Mistral-7B-v0.3 | 512 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| Mistral-7B-v0.3 | 1024 | TurboQuant_INT2_standalone | 79.4311 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| Mistral-7B-v0.3 | 1024 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| Yi-6B | 256 | TurboQuant_INT2_standalone | 139.7776 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| Yi-6B | 256 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| Yi-6B | 512 | TurboQuant_INT2_standalone | 146.7741 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| Yi-6B | 512 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| Yi-6B | 1024 | TurboQuant_INT2_standalone | 210.1293 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| Yi-6B | 1024 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| deepseek-llm-7b-base | 256 | TurboQuant_INT2_standalone | 74.7075 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| deepseek-llm-7b-base | 256 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| deepseek-llm-7b-base | 512 | TurboQuant_INT2_standalone | 68.5732 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| deepseek-llm-7b-base | 512 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| deepseek-llm-7b-base | 1024 | TurboQuant_INT2_standalone | 77.2479 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| deepseek-llm-7b-base | 1024 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| deepseek-llm-7b-base | 1024 | Adaptive_CAREKV_INT3 | 8.8892 | blocked | no |  |
| open_llama_7b_v2 | 256 | TurboQuant_INT2_standalone | 38.4339 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| open_llama_7b_v2 | 256 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| open_llama_7b_v2 | 512 | TurboQuant_INT2_standalone | 41.8835 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| open_llama_7b_v2 | 512 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |
| open_llama_7b_v2 | 1024 | TurboQuant_INT2_standalone | 42.5903 | unstable_outlier_collapse | no | INT2 collapse / extreme PPL degradation |
| open_llama_7b_v2 | 1024 | TurboQuant_plus_CAREKV | — | unsupported | no | QJL is a score-level inner-product estimator, while CARE-KV  |

INT4 rows are in the appendix (`final_quality_appendix_all_rows.csv`) as higher-bit reference.

