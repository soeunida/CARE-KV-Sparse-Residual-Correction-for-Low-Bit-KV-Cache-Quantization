# V / K / both ablation (INT3, S=0.10 R=0.03)

| label | kind | k_scale | v_score | score_normalize | ppl | v_slots_read | k_slots_read | seconds |
|---|---|---|---|---|---|---|---|---|
| v_only | v | 0.000 | output_aware | True | 2.5262 | 90112 | 0 | 230.5 |
| k_only_kscale_0.01 | k | 0.010 | output_aware | True | 2.6148 | 0 | 90112 | 249.4 |
| k_only_kscale_0.02 | k | 0.020 | output_aware | True | 2.6603 | 0 | 90112 | 248.9 |
| k_only_kscale_0.05 | k | 0.050 | output_aware | True | 2.4652 | 0 | 90112 | 249.9 |
| both_kscale_0.01 | both | 0.010 | output_aware | True | 2.7679 | 4227 | 85885 | 278.8 |
| both_kscale_0.02 | both | 0.020 | output_aware | True | 2.7362 | 4149 | 85963 | 278.1 |
| both_kscale_0.05 | both | 0.050 | output_aware | True | 2.6061 | 4201 | 85911 | 277.5 |
