# Absolute budget sweep — INT3 carekv_stored joint+normalize+cached

TinyLlama-1.1B, SEQ_LEN=64, packed_base=True, scale_quant=int8.  READ/STORE_BUDGET_MODE=absolute.  base_quant INT3 PPL = **4.2831**.

| label | store_abs_k | store_abs_v | read_abs_k | read_abs_v | ppl | K_reads | V_reads | mean_delta_K | mean_delta_V | seconds | total_MB | vs_fp16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| invariant_zero | 0 | 0 | 0 | 0 | 4.2831 | 0 | 0 | 0.000e+00 | 0.000e+00 | 1.4 | 0.334 | 0.231 |
| winner_sk4sv4_rk2rv2 | 4 | 4 | 2 | 2 | 3.8294 | 80711 | 99513 | 5.606e-02 | 8.484e-02 | 113.7 | 0.334 | 0.231 |
| K_store_8 | 8 | 4 | 2 | 2 | 3.8294 | 80711 | 99513 | 5.606e-02 | 8.484e-02 | 113.5 | 0.334 | 0.231 |
| V_store_8 | 4 | 8 | 2 | 2 | 3.8294 | 80711 | 99513 | 5.606e-02 | 8.484e-02 | 113.8 | 0.334 | 0.231 |
| k_heavy_4_2 | 4 | 2 | 4 | 2 | 4.0722 | 128602 | 141734 | 6.222e-02 | 5.118e-02 | 99.8 | 0.334 | 0.231 |
| balanced_1 | 1 | 1 | 1 | 1 | 4.0831 | 50116 | 39996 | 4.649e-02 | 2.958e-02 | 76.1 | 0.334 | 0.231 |
| balanced_4 | 4 | 4 | 4 | 4 | 4.1136 | 130275 | 230173 | 5.966e-02 | 8.728e-02 | 115.4 | 0.334 | 0.231 |
| k_heavy_2_1 | 2 | 1 | 2 | 1 | 4.1509 | 87523 | 47645 | 5.694e-02 | 2.886e-02 | 87.3 | 0.334 | 0.231 |
| v_heavy_1_2 | 1 | 2 | 1 | 2 | 4.1883 | 55071 | 80097 | 4.777e-02 | 4.841e-02 | 83.4 | 0.334 | 0.231 |
| balanced_2 | 2 | 2 | 2 | 2 | 4.2200 | 92525 | 87699 | 5.677e-02 | 4.942e-02 | 95.7 | 0.334 | 0.231 |
| v_heavy_2_4 | 2 | 4 | 2 | 4 | 4.2334 | 102558 | 167778 | 5.875e-02 | 8.660e-02 | 111.4 | 0.334 | 0.231 |
| balanced_3 | 4 | 4 | 3 | 3 | 4.2334 | 102558 | 167778 | 5.875e-02 | 8.660e-02 | 117.1 | 0.334 | 0.231 |
| store_rich_read_thin | 4 | 4 | 1 | 1 | 4.3382 | 43338 | 46774 | 4.205e-02 | 6.520e-02 | 109.0 | 0.334 | 0.231 |

## Pareto highlights

- **Best PPL**: `winner_sk4sv4_rk2rv2` (SK=4, SV=4, RK=2, RV=2) → PPL **3.829374** (-0.4537 vs base_quant)
- **Lowest memory** (estimator): `balanced_1` → 0.3337 MB (0.2314× FP16)
- **Best PPL×Memory product**: `winner_sk4sv4_rk2rv2` → PPL 3.829374 × 0.3337 MB
- **Fastest cached run**: `balanced_1` → 76.08 s

## Acceptance checks

1. **R=0 invariant** (SK=SV=RK=RV=0): PPL **4.2831** vs base_quant 4.2831 — **PASS**
2. **Nonzero budgets read slots**: 12/13 cells with positive K_reads or V_reads.
3. **Best both-mode PPL** = **3.8294** — **MATCHES** prior policy-eval winner 3.8294.

## Key observations

- Per-page candidate caps: D/k_channel_group = **2 K** per page, page_size/v_token_block = **4 V** per page.  `SK>2` or `SV>4` is wasted storage (compare `winner_sk4sv4_rk2rv2` vs `K_store_8` — identical PPL).
- At SK=4 SV=4, the **read-budget sweet spot is RK=RV=2**: PPL drops to 3.8294, going to RK=RV=3 (4.23) or RK=RV=4 (4.11) hurts. More reads add noise once the highest-signal slots are already picked.
- Estimator memory shown is dominated by static cache allocation; actual residual-slot usage scales with `stored_K + stored_V` which **caps at 2 K + 4 V per page across the sweep**.
