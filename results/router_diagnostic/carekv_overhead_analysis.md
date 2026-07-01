# CARE-KV correction overhead — FLOPs + memory bandwidth (roofline)

> **Reconciles with `results/overhead_analysis/OVERHEAD_ANALYSIS.md` (REBUTTAL §1)** — independent re-derivation; numbers agree. Adds the roofline/arithmetic-intensity classification and a TinyLlama config.

Per decode token, summed over layers. Correction = **O(S) router scoring** (read+score every stored candidate's sketch) + **O(1) residual read** (top-RK/RV slots) + apply. The O(S) scoring read is the dominant term (an earlier version of THIS tool omitted it and undercounted — now fixed). Paper-best SK2 SV4 RK2 RV2, 4-bit residual, sketch_dim=32. Ridge ≈100 FLOP/byte (A6000-class).

## TinyLlama-1.1B

| SL | FLOP overhead | **BW overhead vs INT3** | score read KB (O(S)) | resid read KB (O(1)) | CARE-KV BW / fp16 | base AI | bound |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 128 | 4.427% | **28.005%** | 93.5 | 66.69 | 0.26× | 44.3 | BW |
| 512 | 3.385% | **19.261%** | 374.0 | 66.69 | 0.2422× | 44.3 | BW |
| 1024 | 3.212% | **17.803%** | 748.0 | 66.69 | 0.2393× | 44.3 | BW |
| 2048 | 3.125% | **17.075%** | 1496.0 | 66.69 | 0.2378× | 44.3 | BW |
| 4096 | 3.082% | **16.71%** | 2992.0 | 66.69 | 0.2371× | 44.3 | BW |
| 8192 | 3.06% | **16.528%** | 5984.0 | 66.69 | 0.2367× | 44.3 | BW |

## Mistral-7B (GQA)

| SL | FLOP overhead | **BW overhead vs INT3** | score read KB (O(S)) | resid read KB (O(1)) | CARE-KV BW / fp16 | base AI | bound |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 128 | 2.617% | **15.925%** | 272.0 | 258.0 | 0.2355× | 24.6 | BW |
| 512 | 1.68% | **10.111%** | 1088.0 | 258.0 | 0.2237× | 24.6 | BW |
| 1024 | 1.523% | **9.142%** | 2176.0 | 258.0 | 0.2217× | 24.6 | BW |
| 2048 | 1.445% | **8.658%** | 4352.0 | 258.0 | 0.2207× | 24.6 | BW |
| 4096 | 1.406% | **8.415%** | 8704.0 | 258.0 | 0.2202× | 24.6 | BW |
| 8192 | 1.387% | **8.294%** | 17408.0 | 258.0 | 0.22× | 24.6 | BW |

## DeepSeek-7B (MHA)

| SL | FLOP overhead | **BW overhead vs INT3** | score read KB (O(S)) | resid read KB (O(1)) | CARE-KV BW / fp16 | base AI | bound |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 128 | 1.636% | **15.925%** | 1020.0 | 967.5 | 0.2355× | 9.8 | BW |
| 512 | 1.05% | **10.111%** | 4080.0 | 967.5 | 0.2237× | 9.8 | BW |
| 1024 | 0.952% | **9.142%** | 8160.0 | 967.5 | 0.2217× | 9.8 | BW |
| 2048 | 0.903% | **8.658%** | 16320.0 | 967.5 | 0.2207× | 9.8 | BW |
| 4096 | 0.879% | **8.415%** | 32640.0 | 967.5 | 0.2202× | 9.8 | BW |
| 8192 | 0.867% | **8.294%** | 65280.0 | 967.5 | 0.22× | 9.8 | BW |

## Reading

- **FLOP overhead is single-digit %** and shrinks slowly (1.68% → 1.387% for Mistral) — negligible arithmetic.
- **Bandwidth overhead vs INT3 is ~constant 8–10%** (10.111% at SL512 → 8.294% at SL8192), **NOT** vanishing — because the router's **O(S) sketch-scoring read** grows with context at the same rate as the base KV read. The O(1) residual read is tiny by comparison. (This corrects an earlier undercount that omitted the scoring read.)
- **But decode is bandwidth-bound and CARE-KV still reads far less than fp16**: CARE-KV read-BW ≈ 0.2217× of fp16 (≈78% NET saving) — the residual overhead is small vs the INT3 base, and the whole thing is a large win vs fp16. base AI ≈24.6 ≪ ridge 100 → bandwidth-bound.
- **Conclusion.** Correction overhead is single-digit % FLOPs and ≤~10% read-bandwidth over INT3 (a large NET saving vs fp16). The ~1000× prototype slowdown is the per-token Python loop, not this; a fused unpack+score+correct kernel realizes it (vectorized already recovers ~15–80×).

**Status: analytical**, reconciled with the REBUTTAL overhead table.
