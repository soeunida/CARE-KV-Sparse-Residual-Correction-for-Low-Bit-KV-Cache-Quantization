# Part E — Residual-computation FLOP & decode-latency overhead

> **Headline**: CARE-KV's residual correction adds only **0.13–0.29% of the
> base attention FLOP** per decode token (at the paper budget SK2/SV4/RK2/RV2:
> **0.27%**), split ~evenly K-correction (36%) / V-correction (36%) / router
> (27%). The overhead is therefore **not** arithmetic — the measured ~**500×**
> decode slowdown (8.5 s/token vs fp16 16.6 ms/token) is pure **Python-loop /
> per-slot gather** overhead. The unblocker is fusion/vectorization, not less
> compute.

## FLOP model (analytical, exact counts)

Per decode token, summed over layers (TinyLlama L=22, Hq=32, Dh=64, SL=1024):

| budget | corr / base FLOP | K share | V share | router share |
|---|---|---|---|---|
| SK1SV2 RK1RV1 | 0.134% | 36% | 36% | 27% |
| **SK2SV4 RK2RV2** (paper) | **0.269%** | 36% | 36% | 27% |
| SK4SV4 RK2RV2 | 0.293% | 33% | 33% | 33% |

Base attention per token ≈ `2·Hq·T·Dh` MACs; correction ≈
`(RK+RV)·Hq·Dh` (K/V) `+ (SK+SV)·sketch·Hq` (router). So
`corr/base ≈ (RK+RV)/(2T)` — **tiny and shrinking with context length**.
CSV: `ablations/carekv_decode_overhead.csv`.

## Lightweight counters (per 64-token decode, RK=RV=2)

q·R_K dot products = **90,112**; V-residual weighted adds = **90,112**;
softmax/Jacobian correction ops = **45,056**; K residual elems read =
5.77 M; V residual elems read = 5.77 M; total residual bytes read ≈ **5.8 MB**.

## Measured wall-clock (`latency/latency.csv`, TinyLlama)

| mode | prefill | decode/token | tok/s |
|---|---|---|---|
| fp16 | 22.8 ms | 16.6 ms | 60.4 |
| base_quant INT3 | 2.0 s | 6.9 s | 0.14 |
| **carekv_stored INT3** | **215.6 s** | **8.5 s** | **0.12** |

## The five reviewer questions

1. **How much extra FLOP does CARE-KV add?** ~**0.27%** of attention FLOP at
   the paper budget (≤0.3% across budgets). Negligible arithmetically.

2. **What dominates the overhead?** In FLOP, K-correction ≈ V-correction
   (36% each) > router (27%). But in *wall-clock* none of these dominate via
   arithmetic — the cost is the per-`(layer, kv_head, token, slot)` Python
   gather/scatter that realizes those few FLOPs.

3. **Is runtime Python-loop bound?** **Yes, overwhelmingly.** carekv decode
   is ~500× slower than fp16 (8.5 s vs 16.6 ms/token) while doing ~0.3% more
   math; base_quant itself is already ~400× slower than fp16 from the
   prototype's Python quant loop. The gap is implementation, not FLOP.

4. **What needs to be fused/vectorized?** The residual **gather + correction**
   (q·R_K dot products and V-residual weighted adds) and the **router slot
   scoring**, currently per-slot Python loops. A batched/Triton kernel that
   (a) gathers the RK/RV selected slots per kv-head and (b) does the q·R and
   weighted-V add as dense matmuls would remove ~all of the overhead, since
   the underlying FLOP is <0.3%. (This matches the CLAUDE.md "vectorized
   joint+both prefill" top-priority item.)

5. **Does overhead grow with seq_len, batch, or budget?** FLOP overhead
   *shrinks* with seq_len (`corr/base ≈ (RK+RV)/2T`), grows linearly with
   read budget (RK+RV) and batch. The *wall-clock* overhead grows with all of
   them because the Python loop is O(slots) = O(L·Hkv·T·(RK+RV)·B) — which is
   exactly why vectorization is the unblocker.

*Diagnostic / prototype-latency. FLOP counts are exact; wall-clock is the
current Python-loop prototype (`latency/latency.csv`), not achievable
kernel runtime.*
