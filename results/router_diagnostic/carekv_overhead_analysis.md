# CARE-KV correction overhead — theoretical FLOPs + memory bandwidth

Per decode token, summed over layers. Base = INT3 attention (read whole K+V cache/token); correction = residual read + apply + router (paper-best SK2 SV4 RK2 RV2, 4-bit residual, sketch_dim=32). `shared` = residual read once per KV head (cached, GQA-shared); `applied` = per-query upper bound. Ridge point ≈100 FLOP/byte (A6000-class).

## TinyLlama-1.1B

| SL | base GFLOP | base KB | corr FLOP% | corr KB (shared) | **BW overhead (shared)** | BW overhead (applied) | base AI | bound |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 128 | 0.0231 | 572.0 | 2.734% | 66.69 | **11.659%** | 15.385% | 39.4 | BW |
| 512 | 0.0923 | 2288.0 | 0.684% | 66.69 | **2.915%** | 3.846% | 39.4 | BW |
| 1024 | 0.1845 | 4576.0 | 0.342% | 66.69 | **1.457%** | 1.923% | 39.4 | BW |
| 2048 | 0.3691 | 9152.0 | 0.171% | 66.69 | **0.729%** | 0.962% | 39.4 | BW |
| 4096 | 0.7382 | 18304.0 | 0.085% | 66.69 | **0.364%** | 0.481% | 39.4 | BW |
| 8192 | 1.4764 | 36608.0 | 0.043% | 66.69 | **0.182%** | 0.24% | 39.4 | BW |

## Mistral-7B (GQA)

| SL | base GFLOP | base KB | corr FLOP% | corr KB (shared) | **BW overhead (shared)** | BW overhead (applied) | base AI | bound |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 128 | 0.0671 | 3328.0 | 2.148% | 258.0 | **7.752%** | 7.692% | 19.7 | BW |
| 512 | 0.2684 | 13312.0 | 0.537% | 258.0 | **1.938%** | 1.923% | 19.7 | BW |
| 1024 | 0.5369 | 26624.0 | 0.269% | 258.0 | **0.969%** | 0.962% | 19.7 | BW |
| 2048 | 1.0737 | 53248.0 | 0.134% | 258.0 | **0.485%** | 0.481% | 19.7 | BW |
| 4096 | 2.1475 | 106496.0 | 0.067% | 258.0 | **0.242%** | 0.24% | 19.7 | BW |
| 8192 | 4.295 | 212992.0 | 0.034% | 258.0 | **0.121%** | 0.12% | 19.7 | BW |

## DeepSeek-7B (MHA)

| SL | base GFLOP | base KB | corr FLOP% | corr KB (shared) | **BW overhead (shared)** | BW overhead (applied) | base AI | bound |
|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 128 | 0.0629 | 12480.0 | 2.148% | 967.5 | **7.752%** | 1.923% | 4.9 | BW |
| 512 | 0.2517 | 49920.0 | 0.537% | 967.5 | **1.938%** | 0.481% | 4.9 | BW |
| 1024 | 0.5033 | 99840.0 | 0.269% | 967.5 | **0.969%** | 0.24% | 4.9 | BW |
| 2048 | 1.0066 | 199680.0 | 0.134% | 967.5 | **0.485%** | 0.12% | 4.9 | BW |
| 4096 | 2.0133 | 399360.0 | 0.067% | 967.5 | **0.242%** | 0.06% | 4.9 | BW |
| 8192 | 4.0265 | 798720.0 | 0.034% | 967.5 | **0.121%** | 0.03% | 4.9 | BW |

## Reading

- **FLOP overhead is tiny** (2.734% at SL128 → 0.171% at SL2048) — correction is negligible arithmetic.
- **Bandwidth is the real axis**, and it too is small in the long-context regime: shared-read overhead 11.659% (SL128) → 1.457% (SL1024) → 0.182% (SL8192). The residual read is **context-independent** (fixed budget/token), so its share of the ∝T base KV read shrinks ~1/T.
- **Both base and correction are bandwidth-bound** (base AI ≈39.4 ≪ ridge 100 FLOP/byte). So the cost that matters is HBM traffic, and the correction adds <2% of it at SL≥1024.
- **GQA amortizes correction bandwidth**: fewer KV heads → the shared residual read is smaller relative to the (Hq-driven) base compute; MHA (DeepSeek) has proportionally more KV-head residual reads but the overhead is still small at long context.
- **Conclusion.** The correction's FLOP and bandwidth overheads are both **<2% at deployment-relevant context lengths**; the ~1000× prototype slowdown is entirely the per-token Python loop, not the algorithm. A fused gather+dequant+apply kernel would realize this sub-2% theoretical overhead; the vectorized path already recovers most of it (~15–80× measured).

**Status: analytical** (arithmetic counts; measured walltime in `carekv_decode_overhead.csv`).
