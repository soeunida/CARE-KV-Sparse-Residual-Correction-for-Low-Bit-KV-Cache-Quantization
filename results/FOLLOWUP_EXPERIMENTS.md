# Follow-up experiments — CARE-KV (Block-GTQ base ⊕ residual)

TinyLlama-1.1B, WikiText-2, Block-GTQ base ⊕ CARE-KV residual, paper-best knobs
(joint+both, packed INT base, int8 scale). `CAREKV_DEBUG_STATS=1` (router validated).

## #1 — KV memory vs sequence length (analytical, `eval_kv_residual_memory_scaling.py`)

CARE-KV INT3 KV cache = **0.2575× fp16**, constant across SL and batch
(≈ 3.9× compression; matches the measured `memory_table.csv` ~0.24–0.26×).
Absolute saving grows with SL×batch:

| batch | SL   | fp16 KV | CARE-KV | saved |
|-------|------|---------|---------|-------|
| 16    | 4096 | 1.375 GB| 0.354 GB| 1.02 GB |
| 16    | 8192 | 2.750 GB| 0.708 GB| 2.04 GB |

→ Unlike model weights (today's batch finding), KV memory is a *pure* per-token
cost, so CARE-KV's win compounds at long context / large batch.
CSV: `results/batch_mem_scaling/kv_memory_scaling.csv`

## #2 — INT2 vs INT3 base × (base_quant, +CARE-KV)  ⭐

| bits | base_quant | +CARE-KV | Δ | improvement |
|------|-----------|----------|-----|-------------|
| INT2 | 19.900 | 17.344 | **−2.556** | **−12.8 %** |
| INT3 | 18.009 | 17.427 | −0.582 | −3.2 % |

Residual correction recovers **4.4× more at INT2** than INT3 — it earns its keep
most where the base quantizer is worst. Notably **INT2+CARE-KV (17.34) ≈
INT3+CARE-KV (17.43)**: the residual lets you drop to a 2-bit base and match the
3-bit-base quality. CSV: `results/int2_carekv/results.csv`

## #3 — cached vs vectorized correction (`eval_prefill_vectorization.py`)  ⚡

| SL  | python | cached | **vectorized** | speedup vs cached | ΔPPL |
|-----|--------|--------|----------------|-------------------|------|
| 64  | 168.3 s| 128.9 s| **2.4 s**  | **54.6×**  | 3.8975→3.8753 |
| 128 | 609.4 s| 842.8 s| **4.1 s**  | **204.6×** | 2.0496→2.0673 |

The vectorized correction is **55–205× faster** than the paper-default `cached`
path, PPL-equivalent (within fp16 noise), and the speedup **grows with SL**.
This directly dissolves the runtime bottleneck measured in the batch sweep
(carekv 527 s/seq under `cached` → ~2–4 s/seq under `vectorized`).
→ Switching the paper-best default to `correction_impl=vectorized` would make
carekv batch/throughput sweeps as cheap as base_quant.
CSV: `results/vectorization_bench/cached_vs_vectorized.csv`

## #4 — Read-budget Pareto (INT3, store fixed SK2 SV4)

| READ (K,V) | PPL | K_reads | V_reads |
|-----------|-----|---------|---------|
| (0,0) base | 18.009 | 0 | 0 |
| (1,1) | **17.332** | 357 628 | 2 820 |
| (2,2) paper | 17.427 | 392 613 | 328 283 |
| (4,4) | 17.220 | 600 815 | 840 977 |

**Non-monotone**: most of the gain comes from a *tiny* read budget — (1,1) already
gets −0.68 PPL with almost no V reads (2 820), and (2,2) is slightly *worse* than
(1,1). Only (4,4) edges ahead. Caveat: N=4 (252 tokens) → the (1,1)/(2,2)/(4,4)
ordering is within noise; the robust takeaway is the sharp (0,0)→(1,1) knee and
diminishing returns after. Consistent with "ratio/read budgets saturate"
(CLAUDE.md §4). CSVs: `results/read_budget_pareto/rk*.csv`

Figure (#2 + #4): `results/followup_experiments.png`
