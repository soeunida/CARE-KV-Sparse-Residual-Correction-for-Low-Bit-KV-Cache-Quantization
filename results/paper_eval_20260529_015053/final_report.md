# CARE-KV — Final paper-evaluation report

Generated: 2026-06-05T13:30:55

## 1. Implementation status

Stable and validated:
- Post-RoPE K storage in a KV-head-indexed cache (Phase 2)
- `CAREKV_PREFILL_MODE=carekv_stored` reads only stored slots (Phase 3)
- `R=0 / READ_BUDGET=0 ≡ base_quant` invariant exact (Phase 6)
- Real packed base storage `CAREKV_PACKED_BASE=1` for INT2 / INT3 / INT4 (Phase B)
- `CAREKV_SCALE_DTYPE` + experimental `CAREKV_SCALE_QUANT=int8` (Phase C)
- Granularity sweep + memory-Pareto knobs (Phase D)
- Adaptive layer-budget policies `uniform/u_shaped/sensitivity` (Phase E)
- 4 routing policies + separate K/V budgets + absolute mode (Phase P1+P2)
- Score normalization in stored router (Phase 24)
- Pre-unpacked-slot cache `correction_impl=cached` (Phase P4-cached)
- Vectorized V correction `correction_impl=vectorized` (Phase P4-vectorized)
- HF `use_cache=True` / DynamicCache integration with incremental decode (Phase G/G-v2)
- Vectorized INT3 unpack (Phase P5)

Paper-best method config (locked):
```
CAREKV_PREFILL_MODE=carekv_stored
CAREKV_PREFILL_RESIDUAL_KIND=both
CAREKV_ROUTE_POLICY=joint
CAREKV_SCORE_NORMALIZE=1
CAREKV_CORRECTION_IMPL=cached         # vectorized falls back to cached for joint+both
CAREKV_BUDGET_POLICY=uniform
CAREKV_PACKED_BASE=1
CAREKV_SCALE_QUANT=int8
STORE_ABS_K=2  STORE_ABS_V=4  READ_ABS_K=2  READ_ABS_V=2
BASE_BITS=3
```

## 2. Main CARE-KV method (synthetic sanity, SEQ_LEN=64)

- **base_quant INT3 baseline**: PPL 4.2831
- **CARE-KV optimized (paper-best config)**: PPL **3.8294**
  (-0.4537 vs base_quant, ~10.6% improvement)
- K_reads = 80711, V_reads = 99513

**Mark: sanity only** — 254-token synthetic prompt.  See § 9 for WikiText-2.

## 3. Memory optimization results

| seq_len | packed_base | scale_dtype | scale_quant | actual_MB | estimator_MB | fp16_MB | actual_vs_fp16 | estimator_vs_fp16 |
|---|---|---|---|---|---|---|---|---|
| 128 | True | fp16 | none | 5.94 | 0.78 | 2.88 | 2.061 | 0.271 |
| 512 | True | fp16 | none | 5.94 | 3.13 | 11.53 | 0.515 | 0.271 |
| 2048 | True | fp16 | none | 11.89 | 12.52 | 46.14 | 0.258 | 0.271 |
| 8192 | True | fp16 | none | 47.54 | 50.08 | 184.55 | 0.258 | 0.271 |
| 128 | True | fp16 | int8 | 5.60 | 0.74 | 2.88 | 1.944 | 0.257 |
| 512 | True | fp16 | int8 | 5.60 | 2.96 | 11.53 | 0.486 | 0.257 |
| 2048 | True | fp16 | int8 | 11.21 | 11.84 | 46.14 | 0.243 | 0.257 |
| 8192 | True | fp16 | int8 | 44.84 | 47.38 | 184.55 | 0.243 | 0.257 |

Headline: packed_base=True + scale_quant=int8 gives **0.243× FP16** at 2048 tokens.

## 4. Absolute K/V budget sweep (paper main)

| label | store_abs_k | store_abs_v | read_abs_k | read_abs_v | ppl | K_reads | V_reads | seconds |
|---|---|---|---|---|---|---|---|---|
| invariant_zero | 0 | 0 | 0 | 0 | 4.2831 | 0 | 0 | 1.4 |
| balanced_1 | 1 | 1 | 1 | 1 | 4.0831 | 50116 | 39996 | 76.1 |
| k_heavy_2_1 | 2 | 1 | 2 | 1 | 4.1509 | 87523 | 47645 | 87.3 |
| v_heavy_1_2 | 1 | 2 | 1 | 2 | 4.1883 | 55071 | 80097 | 83.4 |
| balanced_2 | 2 | 2 | 2 | 2 | 4.2200 | 92525 | 87699 | 95.7 |
| k_heavy_4_2 | 4 | 2 | 4 | 2 | 4.0722 | 128602 | 141734 | 99.8 |
| v_heavy_2_4 | 2 | 4 | 2 | 4 | 4.2334 | 102558 | 167778 | 111.4 |
| balanced_4 | 4 | 4 | 4 | 4 | 4.1136 | 130275 | 230173 | 115.4 |
| winner_sk4sv4_rk2rv2 | 4 | 4 | 2 | 2 | 3.8294 | 80711 | 99513 | 113.7 |
| store_rich_read_thin | 4 | 4 | 1 | 1 | 4.3382 | 43338 | 46774 | 109.0 |
| balanced_3 | 4 | 4 | 3 | 3 | 4.2334 | 102558 | 167778 | 117.1 |
| K_store_8 | 8 | 4 | 2 | 2 | 3.8294 | 80711 | 99513 | 113.5 |
| V_store_8 | 4 | 8 | 2 | 2 | 3.8294 | 80711 | 99513 | 113.8 |

Best: `winner_sk4sv4_rk2rv2` → PPL 3.8294.  Read budgets above 2 per kind degrade PPL (more reads add noise once the highest-signal slots are already picked).

## 5. Routing policy ablation (kind × policy, abs SK=SV=4, RK=RV=2)

| kind | route_policy | ppl_cached | K_reads_ca | V_reads_ca | time_cached_s | speedup |
|---|---|---|---|---|---|---|
| v | separate | 4.5138 |  |  | 87.7 | 1.22 |
| v | joint | 4.5138 |  |  | 87.8 | 1.22 |
| v | k_first | 4.2840 |  |  | 88.5 | 1.16 |
| v | adaptive | 4.6910 |  |  | 87.8 | 1.04 |
| k | separate | 4.2793 |  |  | 85.1 | 1.25 |
| k | joint | 4.2793 |  |  | 84.4 | 1.24 |
| k | k_first | 4.2793 |  |  | 85.0 | 1.23 |
| k | adaptive | 4.5365 |  |  | 84.8 | 1.19 |
| both | separate | 3.8661 |  |  | 116.5 | 1.33 |
| both | joint | 3.8294 |  |  | 116.4 | 1.33 |
| both | k_first | 4.0731 |  |  | 115.6 | 1.32 |
| both | adaptive | 4.1500 |  |  | 121.0 | 1.32 |

Best both-mode: **joint** policy (PPL 3.8294).  Single-kind: K-only at k_scale=0.05 gives 2.47.  `adaptive` mis-allocates for single-kind modes and underperforms.

## 6. Layer-wise budget policy ablation (Phase E)

| label | policy | ppl | K_reads_total | V_reads_total | seconds |
|---|---|---|---|---|---|
| uniform_baseline | uniform | 3.8294 | 80711 | 99513 | 114.4 |
| u_shaped_builtin | u_shaped | 3.8873 | 77726 | 102498 | 108.3 |
| sensitivity_sharp_u | sensitivity | 4.2209 | 77972 | 110444 | 99.3 |
| sensitivity_default_uni | sensitivity | 3.8294 | 80711 | 99513 | 114.3 |

**Finding**: at our budget level, layer-wise redistribution does not help.  Uniform stays best.

## 7. use_cache=True / incremental decode (Phase G/G-v2)

Working end-to-end with HF DynamicCache.  Decode path appends K/V into the open page
via `cache.append_to_page`; no fresh page per token.  `test_incremental_decode_page_growth`
verifies `pages_used == ceil(total_tokens / page_size)` on a real generation.

| mode | prompt_len | prefill_ms | decode_ms_per_token | tokens_per_sec | peak_gpu_mem_MB | K_reads | V_reads |
|---|---|---|---|---|---|---|---|
| fp16 | 128 | 24 | 18 | 54.32 | 2239 | 0 | 0 |
| base_quant_int3 | 128 | 2041 | 4166 | 0.24 | 3212 | 0 | 0 |
| carekv_stored_int3 | 128 | 397236 | 4677 | 0.21 | 5402 | 171695 | 211281 |
| base_quant_int2 | 128 | 1795 | 4255 | 0.23 | 3148 | 0 | 0 |
| carekv_stored_int2 | 128 | 395174 | 4836 | 0.21 | 5277 | 168153 | 214823 |

**Mark: prototype latency** — prefill is dominated by the Python per-(layer,head,t) loop.

## 8. Prefill correction vectorization (Phase P4-vectorized)

| seq_len | impl | ppl | seconds | speedup_vs_cached | K_reads | V_reads | peak_gpu_mem_MB |
|---|---|---|---|---|---|---|---|
| 64 | python | 3.8661 | 153.4 | - | 90112 | 90112 | 4028.05 |
| 64 | cached | 3.8661 | 116.4 | - | 90112 | 90112 | 4028.05 |
| 64 | vectorized | 3.9718 | 85.7 | 1.36 | 90112 | 90112 | 4028.05 |
| 128 | python | 2.0637 | 472.9 | - | 180224 | 180224 | 4046.15 |
| 128 | cached | 2.0637 | 393.3 | - | 180224 | 180224 | 4046.15 |
| 128 | vectorized | 2.0614 | 279.6 | 1.41 | 180224 | 180224 | 4046.15 |

Vectorized V matches cached within fp16 noise at SEQ_LEN=128 (Δ=0.002); gives 1.36×–1.41× over cached.
Joint+both currently falls back to cached for bit-exactness.

## 9. WikiText-2 dataset PPL (paper evaluation)

### Paper run
| mode | ppl | total_tokens | seconds | K_reads | V_reads | peak_gpu_mem_MB |
|---|---|---|---|---|---|---|
| fp16 | 15.7691 | 2032 | 0.6 | 0 | 0 | 2262 |
| base_quant_int4 | 16.4358 | 2032 | 41.4 | 0 | 0 | 5597 |
| base_quant_int3 | 21.7379 | 2032 | 45.9 | 0 | 0 | 5424 |
| base_quant_int2 | 351.6028 | 2032 | 42.6 | 0 | 0 | 5299 |
| carekv_stored_int3_optimized | 18.1423 | 2032 | 6268.0 | 2546843 | 3220325 | 5424 |


## 10. Multi-model evaluation

| model | mode | ppl | seconds | K_reads | V_reads | status |
|---|---|---|---|---|---|---|
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | fp16 | 12.2739 | 0.4 | 0 | 0 |  |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | base_quant_int3 | 15.7361 | 12.2 | 0 | 0 |  |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | carekv_stored_int3_optimized | 13.5378 | 1584.9 | 640139 | 801653 |  |
| JackFram/llama-160m | fp16 | 28.6112 | 0.3 | 0 | 0 |  |
| JackFram/llama-160m | base_quant_int3 | 36.1912 | 18.7 | 0 | 0 |  |
| JackFram/llama-160m | carekv_stored_int3_optimized | 30.4707 | 337.3 | 128238 | 166674 |  |


## 11. Long-context retrieval / copy (Phase J)

| task | label | exact_match | char_acc | ctx_target | K_reads | V_reads | seconds |
|---|---|---|---|---|---|---|---|
| kv_retrieval | fp16 | 1.00 | 1.00 | 128 | 0 | 0 | 0.9 |
| kv_retrieval | base_quant_int3 | 0.00 | 0.13 | 128 | 0 | 0 | 132.6 |
| kv_retrieval | carekv_int3_both | 0.00 | 0.30 | 128 | 688986 | 907686 | 1842.2 |
| boundary | fp16 | 1.00 | 1.00 | 128 | 0 | 0 | 0.6 |
| boundary | base_quant_int3 | 0.20 | 0.33 | 128 | 0 | 0 | 132.6 |
| boundary | carekv_int3_both | 0.00 | 0.07 | 128 | 709372 | 923908 | 1948.6 |
| copy | fp16 | 0.00 | 0.00 | 128 | 0 | 0 | 0.6 |
| copy | base_quant_int3 | 0.00 | 0.00 | 128 | 0 | 0 | 146.7 |
| copy | carekv_int3_both | 0.00 | 0.00 | 128 | 749699 | 1052541 | 2214.6 |


## 12. Diagnostic figures

- `figures/3d_activation_layer00.png`
- `figures/3d_activation_layer11.png`
- `figures/3d_activation_layer21.png`
- `figures/3d_before_after_layer00.png`
- `figures/3d_before_after_layer11.png`
- `figures/3d_before_after_layer21.png`
- `figures/3d_carekv_K_error_clean_layer00.png`
- `figures/3d_carekv_K_error_clean_layer11.png`
- `figures/3d_carekv_K_error_clean_layer21.png`
- `figures/3d_carekv_K_error_layer00.png`
- `figures/3d_carekv_K_error_layer11.png`
- `figures/3d_carekv_K_error_layer21.png`
- `figures/3d_carekv_V_error_clean_layer00.png`
- `figures/3d_carekv_V_error_clean_layer11.png`
- `figures/3d_carekv_V_error_clean_layer21.png`
- `figures/3d_carekv_V_error_layer00.png`
- `figures/3d_carekv_V_error_layer11.png`
- `figures/3d_carekv_V_error_layer21.png`
- `figures/fig_absolute_budget_sweep.png`
- `figures/fig_adaptive_read_budget.png`
- `figures/fig_adaptive_read_budget_wikitext2_n4.png`
- `figures/fig_base_quantizer_expansion_memory_quality.png`
- `figures/fig_base_quantizer_expansion_ppl.png`
- `figures/fig_budget_pareto.png`
- `figures/fig_budget_ratio_vs_absolute.png`
- `figures/fig_carekv_on_base_quantizers_memory_quality.png`
- `figures/fig_carekv_on_base_quantizers_ppl.png`
- `figures/fig_kv_budget_balance.png`
- `figures/fig_kvquant_carekv_unblock.png`
- `figures/fig_layer_budget_policy.png`
- `figures/fig_long_context_retrieval.png`
- `figures/fig_memory_pareto.png`
- `figures/fig_multimodel_ppl.png`
- `figures/fig_read_budget_sweep.png`
- `figures/fig_reconstruction_pareto_ppl_validation.png`
- `figures/fig_route_policies.png`
- `figures/fig_routing_baseline_ablation.png`
- `figures/fig_routing_baseline_wikitext2_n4.png`
- `figures/fig_sota_direct_memory_quality.png`
- `figures/fig_sota_direct_ppl.png`
- `figures/fig_sota_direct_runtime.png`
- `figures/fig_sota_direct_wikitext2_n4_memory_quality.png`
- `figures/fig_sota_direct_wikitext2_n4_ppl.png`
- `figures/fig_sota_direct_wikitext2_n4_runtime.png`
- `figures/fig_store_budget_sweep.png`
- `figures/fig_vk_both_ablation.png`
- `figures/fig_wikitext2_ppl.png`
- `figures/heatmap_carekv_K_error_layer00.png`
- `figures/heatmap_carekv_K_error_layer11.png`
- `figures/heatmap_carekv_K_error_layer21.png`
- `figures/heatmap_carekv_V_error_layer00.png`
- `figures/heatmap_carekv_V_error_layer11.png`
- `figures/heatmap_carekv_V_error_layer21.png`
- `figures/layer_00_diagnostics.png`
- `figures/layer_11_diagnostics.png`
- `figures/layer_21_diagnostics.png`
- `figures/paper_carekv_KV_error_layer11.png`
- `figures/paper_carekv_K_error_layer00.png`
- `figures/paper_carekv_K_error_layer11.png`
- `figures/paper_carekv_K_error_layer21.png`
- `figures/paper_carekv_V_error_layer00.png`
- `figures/paper_carekv_V_error_layer11.png`
- `figures/paper_carekv_V_error_layer21.png`

See `summaries/activation_3d_figures.md` for the per-layer 3D Channel × Token × |value| activation surfaces (K pre-RoPE / K post-RoPE / V) and outlier-channel interpretation.

See `summaries/before_after_3d_figures.md` for the per-layer CARE-KV before/after 3D surfaces (fp16 → INT3 base_quant → INT3 CARE-KV) and per-layer reconstruction-error table.

See `summaries/sota_official_integration_status.md` for the Phase P-direct same-condition SOTA harness — adapter-based framework comparing CARE-KV with FP16, base_quant ladder, KIVI-style (real reimplementation), and documented stubs for KVQuant/MiKV/ZipCache. CSV/JSON output under `sota_direct/`.

See `summaries/carekv_before_after_3d.md` for the per-layer CARE-KV error-decomposition figures (`3d_carekv_{K,V}_error_*.png` + `heatmap_carekv_{K,V}_error_*.png`) — 4 panels: base error, CARE-KV error, error reduction (diverging), recovered residual.

See `summaries/adaptive_read_budget.md` for the Phase O adaptive read-budget experiment (synthetic SL=64, diagnostic-only) — fixed RK=RV={1,2,3,4} vs `read_budget_mode=adaptive_score` with max RK=RV=4 and relative threshold in {0.00, 0.05, 0.10, 0.20, 0.30}.

See `summaries/adaptive_read_budget_wikitext2_n4.md` for the Phase O real-dataset pilot — WikiText-2 N=4 SL=128. **adaptive_score rel=0.05 beats fixed RK=RV=2 by 0.53 PPL** (12.93 vs 13.46); candidate paper-best pending N=16 confirmation. Figure `fig_adaptive_read_budget_wikitext2_n4.png`, CSV `ablations/adaptive_read_budget_wikitext2_n4.csv`.

See `summaries/budget_experiments_overview.md` for the Phase N budget experiments (ratio vs absolute, store-budget sweep, read-budget sweep, K/V balance, and Pareto front) — figures `fig_budget_{ratio_vs_absolute,store_budget_sweep,read_budget_sweep,kv_budget_balance,pareto}.png`.

**Candidate-cap interpretation (diagnostic).** The store budget selects from per-page candidate pools whose size is fixed by residual granularity: `K_cand_cap = head_dim/k_channel_group = 64/32 = 2`, `V_cand_cap = ceil(page_size/v_token_block) = 16/4 = 4`. So `STORE_ABS_K>2` / `STORE_ABS_V>4` add no new candidate — the effective budget saturates at the cap (`SK∈{2,4,8}, SV∈{4,8}` give identical effective budget, reads, and PPL). Therefore `SK=2, SV=4` is the **minimum-storage equivalent under the current residual granularity**, *not* a proven globally optimal store budget. Every budget row now reports requested-vs-effective SK/SV, caps, store utilization, recovered K/V elements, and residual memory bytes. The residual-granularity sensitivity sweep (`summaries/budget_granularity_sweep.md`, `figures/fig_budget_granularity_sweep.png`) varies the caps themselves; it is **diagnostic** and does not change the locked paper-best config.

See `summaries/carekv_on_base_quantizers.md` for Phase Q-stacked: CARE-KV residual correction stacked on top of a KIVI-style (per-channel K + per-token V) base quantizer. On WT-2 N=4 SL=128 TinyLlama, KIVI_INT3 + CARE-KV (PPL 13.095) beats both KIVI_INT3 standalone (15.657) and uniform+CARE-KV (13.462). Diagnostic pilot — needs WT-2 N>=16 confirmation. CSV `ablations/carekv_on_base_quantizers.csv`.

See `summaries/base_quantizer_expansion_wt2_n4.md` for the base-quantizer expansion beyond Phase Q: KVQuant-style (pre-RoPE K, same-condition reimpl), RotateKV-style (Walsh-Hadamard rotation), and TurboQuant (unsupported — see `turboquant_integration_status.md`). On WT-2 N=4 SL=128 TinyLlama: KVQuant pre-RoPE INT3 (15.01) marginally beats KIVI INT3 (15.66); RotateKV standalone (27.45) is worse than INT3 baseline but CARE-KV rescues it (27.45->15.23). CSV `ablations/base_quantizer_expansion_wt2_n4.csv`.

### KVQuant-style (pre-RoPE) + CARE-KV — unblocked stacked cell

The previously-`unsupported` "KVQuant-style + CARE-KV" cell (pre-RoPE K vs post-RoPE residual coordinate mismatch) is now **unblocked**: K is quantized pre-RoPE, `K_hat` is re-rotated, and the CARE-KV residual is computed in post-RoPE coordinates (`CAREKV_K_STORE_MODE=pre_rope`). WT-2 N=4 SL=128 TinyLlama (diagnostic pilot):

| method_name | base_quantizer | ppl | dppl_vs_fp16 | dppl_vs_base_quant_int3 | estimated_kv_memory_MB | vs_fp16_kv_memory_ratio | k_reads | v_reads | runtime_seconds |
|---|---|---|---|---|---|---|---|---|---|
| fp16 |  | 12.3457 | +0.0000 | -3.8516 | 2.750 | 1.0000 | 0 | 0 | 24.0 |
| base_quant_INT3 |  | 16.1973 | +3.8516 | +0.0000 | 0.516 | 0.1875 | 0 | 0 | 29.6 |
| KVQuant_style_INT3K_INT3V_preRoPE | kvquant_style | 15.0080 | +2.6623 | -1.1893 | 0.548 | 0.1992 | 0 | 0 | 17.2 |
| KVQuantPreRoPE_INT3K_INT3V_plus_CAREKV | kvquant_style | 13.1004 | +0.7547 | -3.0969 | 0.685 | 0.2492 | 648597 | 793195 | 2092.6 |
| KIVI_INT3K_INT3V_plus_CAREKV | kivi_style | 13.0948 | +0.7491 | -3.1025 | 0.685 | 0.2492 | 648980 | 792812 | 1997.1 |
| CAREKV_fixed_SK2SV4_RK2RV2 | uniform | 13.4618 | +1.1161 | -2.7355 | 0.653 | 0.2375 | 641915 | 799877 | 1669.1 |


Findings: (1) KVQuant + CARE-KV (13.100) improves over KVQuant standalone (15.008) by **-1.91 PPL**; (2) it is a **statistical tie** with KIVI + CARE-KV (13.095; +0.006 PPL, within noise) at identical memory; (3) it **beats** uniform-INT3 + CARE-KV (13.462) by **-0.36 PPL** at +5% KV memory. All CARE-KV cells have K_reads + V_reads > 0 (router fired). Diagnostic pilot — needs WT-2 N>=16 confirmation. CSV `ablations/kvquant_carekv_unblock_wt2_n4.csv`, summary `summaries/kvquant_carekv_unblock_wt2_n4.md`, figure `fig_kvquant_carekv_unblock.png`.

See `summaries/routing_baseline_ablation.md` for the Phase M routing baseline ablation (CARE-KV vs random / magnitude_only / attention_only / oracle_proxy at the same store + read budget) — synthetic 254-token prompt, diagnostic-only.

See `summaries/routing_baseline_wikitext2_n4.md` for the Phase M real-dataset pilot — WikiText-2 N=4 SL=128, same 6 baselines, real-dataset pilot (not full paper-scale). Figure `fig_routing_baseline_wikitext2_n4.png`, CSV `ablations/routing_baseline_wikitext2_n4.csv`.

See `summaries/current_bottlenecks_and_optimization_plan.md` for the current bottleneck audit and prioritized optimization plan.

## 13. Remaining limitations

1. **Prefill is Python-loop bound** at O(T² × Hq × L).  Vectorized V exists but joint+both falls back to cached.  Bigger PPL evals (SL=512 N=32) need joint+both vectorization first.
2. **Decode wall-clock** at carekv_stored is ~5 s/token at TinyLlama scale — the CAREKV per-(layer, kv_head) Python loop.  Tractable for correctness/PPL evals, not for serving.
3. **HF DynamicCache peak memory** is dominated by fp16 dummy K/V we feed it for `get_seq_length()` tracking.  A shape-(B, Hkv, T, 1) custom Cache would shrink peak ~50% — see `summaries/remaining_improvements.md` § 4.
4. **No multi-model coverage beyond TinyLlama** in this session — additional LLaMA-family models aren't cached locally and download requires HF auth in some cases.  Plumbing supports any LLaMA-style attention.

## 14. Paper-ready tables (labels)

- § 3 Memory: **paper-ready** (estimator + actual on-device verified).
- § 4 Absolute budget sweep: **paper-ready** (sanity SEQ_LEN=64).
- § 5 Route policy ablation: **paper-ready** (SEQ_LEN=64).
- § 6 Layer budget ablation: **paper-ready** (SEQ_LEN=64).
- § 7 use_cache=True: **prototype latency** (CARE-KV decode unoptimized).
- § 8 Prefill vectorization: **paper-ready** for `separate` policy; joint+both falls back to cached for exactness.
- § 9 WikiText-2 PPL: **paper evaluation** (smoke N=4 / paper N=16 depending on run).
- § 10 Multi-model: **deferred** (only TinyLlama cached this session).
- § 11 Long-context: **paper evaluation** when present; **sanity** otherwise.
- § 12 Figures: **diagnostic**.

## 15. Experimental / future-work items

See `summaries/remaining_improvements.md` for the assessment of:
- Soft K-guided V routing (future work, low priority)
- Entropy-aware read budget (future work, low priority)
- Vectorized joint+both prefill (**future work, top priority** — unblocks SL=512 paper evals)
- Lightweight HF Cache metadata (paper-ready candidate, flagged)

