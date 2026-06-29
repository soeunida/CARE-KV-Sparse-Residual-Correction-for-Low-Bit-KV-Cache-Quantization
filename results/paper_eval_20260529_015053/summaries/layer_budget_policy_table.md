# Phase E — adaptive layer-wise budget policies

TinyLlama-1.1B, SEQ_LEN=64, INT3 / packed_base / scale_quant=int8 / joint+normalize / cached.
Base global budget: **SK=2, SV=4, RK=2, RV=2** (the absolute-sweep winner).
Per-layer multiplier is mean-normalized across the 22 layers so total budget across the network is preserved.

| label | policy | PPL | ΔPPL vs uniform | K_reads | V_reads | seconds |
|---|---|---:|---:|---:|---:|---:|
| **uniform_baseline** | uniform | **3.8294** | 0.0000 | 80,711 | 99,513 | 114.4 |
| u_shaped_builtin | u_shaped | 3.8873 | +0.0579 | 77,726 | 102,498 | 108.3 |
| sensitivity_sharp_u | sensitivity | 4.2209 | +0.3915 | 77,972 | 110,444 | 99.3 |
| sensitivity_default_uni | sensitivity | 3.8294 | 0.0000 | 80,711 | 99,513 | 114.3 |

## Per-layer K stored slots (22 layers)

| label | layer-by-layer stored_K |
|---|---|
| uniform | `[32]×22` |
| u_shaped | `[32, 32, 32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32]` |
| sensitivity_sharp_u | `[32, 32, 32, 32, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32]` |

(K slots are stored per (KV head × page), with 4 KV heads × ~4 pages × 2 K candidates per page = 32 per layer at uniform.)

## Per-layer multiplier profiles

| label | profile |
|---|---|
| uniform | `1.00` for all 22 layers |
| u_shaped (built-in) | `1.56, 1.44, 1.33, 1.22, 1.11, 1.00, 0.89, 0.78, 0.67, 0.56, 0.44, 0.44, 0.56, 0.67, 0.78, 0.89, 1.00, 1.11, 1.22, 1.33, 1.44, 1.56` |
| sensitivity (sharp U: 4 edge / 14 mid / 4 edge with weights 2/0.5/2) | `1.91×4, 0.48×14, 1.91×4` |

## Acceptance

1. **uniform reproduces the absolute-sweep best PPL=3.8294** exactly ✓
2. **u_shaped and sensitivity both ran successfully** ✓
3. **Neither adaptive policy improved on uniform** at this configuration (TinyLlama-1.1B, SEQ_LEN=64, SK=2 SV=4 RK=2 RV=2):
   - u_shaped (gentle 1.56/0.44) → +0.058 PPL
   - sensitivity (sharp 1.91/0.48) → +0.392 PPL
4. The **sensitivity-with-default-uniform-weights** row reproduces the uniform PPL exactly (3.8294), confirming that the sensitivity policy correctly degenerates to uniform when `cfg.layer_sensitivity` is all-ones.

## Honest interpretation

For TinyLlama-1.1B at this budget level, **mid-layer residual correction is *not* less valuable than edge-layer correction**.  The classical "U-shaped sensitivity" intuition (early + late layers matter most) does not transfer at the slot-routing granularity tested here.  The sharper the U, the more PPL degrades.

Possible reasons:
- TinyLlama's per-layer error attribution may be flatter than the U-shape literature suggests for larger models.
- At this budget (2 reads per query per kind), every layer is already close to its own marginal-utility ceiling; redistributing among layers can only break a working balance.
- A learned-static calibration (run base_quant once per layer with correction added and measure ΔPPL → use as sensitivity weights) might reveal a non-U profile that *does* help.  Future work.

**Paper recommendation**: keep `budget_policy=uniform` as the main carekv_stored result.  Report u_shaped/sensitivity as ablations that did not improve, showing that the gains in this work come from per-kind separated budgets + score normalization + the right read-budget sweet spot — *not* from layer-wise allocation tricks.
