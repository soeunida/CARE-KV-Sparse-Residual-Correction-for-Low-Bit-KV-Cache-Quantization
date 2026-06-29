# Core PPL (TinyLlama-1.1B-Chat-v1.0)

Labels:  fp = full precision reference · base_quant = low-bit base KV only · carekv_eval = full-residual upper bound · carekv_stored = paper-quality stored-slot path.

| label | base_bits | prefill_mode | store_budget | read_budget | kind | k_scale | ppl | v_slots_read | k_slots_read | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| fp | 16 | fp | 0.00 | 0.00 | - | 0.000 | 2.2916 | 0 | 0 | 0.3 |
| base_quant_int4 | 4 | base_quant | 0.00 | 0.00 | both | 0.100 | 2.3562 | 0 | 0 | 1.9 |
| base_quant_int3 | 3 | base_quant | 0.00 | 0.00 | both | 0.100 | 2.6903 | 0 | 0 | 2.0 |
| base_quant_int2 | 2 | base_quant | 0.00 | 0.00 | both | 0.100 | 223.6506 | 0 | 0 | 1.8 |
| carekv_eval_int3_v_R005 | 3 | carekv_eval | 0.10 | 0.05 | v | 0.100 | 2.4948 | 0 | 0 | 33.3 |
| carekv_stored_int3_v | 3 | carekv_stored | 0.10 | 0.03 | v | 0.100 | 2.5262 | 90112 | 0 | 228.6 |
| carekv_stored_int2_v | 2 | carekv_stored | 0.20 | 0.05 | v | 0.100 | 208.7451 | 90112 | 0 | 229.5 |
| carekv_stored_int3_both | 3 | carekv_stored | 0.10 | 0.03 | both | 0.050 | 2.5749 | 49 | 90063 | 278.1 |
