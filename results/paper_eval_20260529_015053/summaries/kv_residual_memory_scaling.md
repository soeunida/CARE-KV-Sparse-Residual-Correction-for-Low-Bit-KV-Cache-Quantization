# Part C — KV memory vs CARE-KV residual memory scaling (analytical)

> **Headline**: At GB-scale KV (Qwen2.5-7B, batch=8, seq=2048 → **0.875 GB
> fp16 KV**), CARE-KV INT3 (SK2/SV4) lands at **0.20 GB = 0.23× fp16**, and
> the residual+metadata is only **~9% of the CARE-KV total** (residual slots
> alone ~5.8%). Residual memory does **not** erase the quantization saving.

## Method

Pure memory **estimator** (no model forward) sweeping
`model × batch{1,2,4,8} × seq_len{128,512,1024,2048} × bits{4,3,2} ×
budget{SK1SV2, SK2SV4, SK4SV4}` = 576 rows. Per-component byte model is
re-derived from the project's measured `memory/memory_table.csv`
decomposition (base_code / scale / residual / meta / sketch):

- packed base INT-b code: `head_dim·b/8` bytes per (layer,kv_head,token,{K,V})
- int8 page scale: `1/32` byte/elem
- residual slot: **1.13 B/slot** (≈4-bit value + ~4-bit local index)
- metadata: **0.524 B/slot** (residual index/bookkeeping)
- router sketch: **4.0 B/token** per (layer,kv_head) (sketch_dim=16 int8)

`stored_slots = L·Hkv·(B·S)·(SK+SV)`. CSV:
`ablations/kv_residual_memory_scaling.csv`, figure
`fig_kv_residual_memory_scaling.png`.

Models: **Qwen2.5-7B** (28L, 4 kv-heads, d=128 — GQA), **TinyLlama-1.1B**
(22L, 4 kv-heads, d=64 — GQA), and a projected **LLaMA-7B-MHA** (32L, 32
kv-heads, d=128) to show the GQA effect.

## The five reviewer questions

**1. At what batch/context does KV reach GB scale?**
Depends sharply on GQA. Per-token fp16 KV:
- Qwen2.5-7B (4 kv-heads): **56 KB/token** → 1 GB at ~18.7K tokens
  (e.g. B=8×S=2048 = 16.4K → 0.875 GB; B=8×S=2304 ≈ 1 GB).
- LLaMA-7B-**MHA** (32 kv-heads, projected): **448 KB/token** → 1 GB at
  only ~2.3K tokens (B=2×S=1024 already 0.875 GB). GQA delays GB-scale by
  the group factor (≈8×).
- TinyLlama (4 kv-heads, d=64): 11 KB/token — sub-GB across the whole grid.

**2. Does residual memory dominate or stay small?**
**Stays small.** At the paper budget SK2/SV4, residual slots are
**~5.8% of the CARE-KV total** and residual+metadata+sketch together ~9%,
roughly constant across batch/seq (both residual and base scale linearly in
tokens). Even the largest budget swept (SK4/SV4) keeps residual under ~9%.

**3. Does CARE-KV still save memory after residuals?**
**Yes.** Qwen2.5-7B INT3 totals (× fp16): SK1SV2 **0.213×**, SK2SV4
**0.230×**, SK4SV4 **0.245×**. I.e. CARE-KV still saves **75–79%** of fp16
KV *after* counting residual+metadata+sketch. INT4 budgets land ~0.28–0.30×.

**4. Which budget has the best PPL/memory trade-off?**
On memory alone the budgets are within ~3% of each other (residual is a
small fraction), so the budget choice is dominated by **quality**, not
memory. From the paper ablations the SK2/SV4/RK2/RV2 point is the
quality/memory sweet spot (this part is memory-only; PPL is Parts A/B/D).
Because residual memory is nearly free, **spending more store budget is
cheap** — the constraint is read budget / latency, not memory.

**5. Does BaseQuant INT4 already solve the problem at this scale?**
For **memory**, BaseQuant INT4 reaches ~0.28× fp16 (vs CARE-KV INT3
~0.23×) — both solve the GB→sub-GB problem. So at the memory axis INT4 is
"good enough". The remaining question is **quality at low bits**: CARE-KV's
value is delivering INT3-or-below memory at INT4-or-better quality (Parts
A/B/D), which BaseQuant INT4 does not address. Memory scaling alone does
**not** settle the INT4-vs-CARE-KV question — quality does.

## Key numbers (Qwen2.5-7B, GQA)

| B×S (tokens) | fp16 KV | BaseQuant INT4 | CARE-KV INT3 (SK2SV4) | residual % of CARE total |
|---|---|---|---|---|
| 1×2048 (2K) | 0.109 GB | 0.031 GB | 0.025 GB | 5.8% |
| 8×1024 (8K) | 0.438 GB | 0.123 GB | 0.101 GB | 5.8% |
| 8×2048 (16K) | 0.875 GB | 0.246 GB | 0.201 GB | 5.8% |

*Estimator (no forward). GQA (4 kv-heads) is what keeps Qwen-7B KV
sub-GB until ~16K tokens; an MHA-7B would hit GB at ~2K tokens.*
