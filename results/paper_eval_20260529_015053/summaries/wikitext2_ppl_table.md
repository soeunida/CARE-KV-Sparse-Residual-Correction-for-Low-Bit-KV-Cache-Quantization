# WikiText-2 PPL (Phase I)

TinyLlama-1.1B-Chat-v1.0, `wikitext-2-raw-v1` test split.
SEQ_LEN=128, NUM_SAMPLES=4 (508 evaluated tokens, smoke run).
carekv_stored uses the paper-best config: joint+normalize+cached, packed_base=1, scale_quant=int8, abs SK=2 SV=4 RK=2 RV=2, uniform.

| mode | PPL | ΔPPL vs fp16 | recovers vs base_quant | tokens | seconds | K_reads | V_reads | peak_MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 12.2739 | +0.0000 | - | 508 | 0.35 | 0 | 0 | 2261.58 |
| base_quant_int4 | 12.6628 | +0.3890 | - | 508 | 10.7 | 0 | 0 | 5597.37 |
| base_quant_int3 | 15.7361 | +3.4622 | - | 508 | 11.96 | 0 | 0 | 5424.48 |
| base_quant_int2 | 244.5758 | +232.3020 | - | 508 | 10.98 | 0 | 0 | 5298.52 |
| carekv_stored_int3_optimized | 13.5378 | +1.2639 | **63%** of INT3→fp16 gap | 508 | 1560.95 | 640139 | 801653 | 5424.48 |

## Headline finding

INT3 base_quant on WikiText-2 hits PPL **15.74**, **+3.46** vs fp16.
**CARE-KV optimized INT3** recovers to PPL **13.54**, **+1.26** vs fp16 — closing **63%** of the INT3→fp16 quantization gap with only an estimated 24-26% of the FP16 KV-cache memory.

- INT4 base_quant for reference is +0.39 PPL.  CARE-KV at INT3 (~25% memory) is at +1.27 PPL — slightly worse than INT4 but at much lower memory.
- INT2 base_quant is unusable on WikiText-2 (PPL 244+); carekv_stored INT2 was excluded from this smoke run to keep runtime tractable (Python prefill loop, ~26 min for INT3).

## Runtime caveats

- carekv_stored INT3 took 1561 s for 4 windows (≈ 6 min/window).
- K_reads=640139 and V_reads=801653 confirm the optimized routing is firing across all 22 layers and 32 query heads.
- Scaling to NUM_SAMPLES=32 / SEQ_LEN=512 (the paper-target eval) would need a vectorized joint+both prefill, which is the next major optimization.
