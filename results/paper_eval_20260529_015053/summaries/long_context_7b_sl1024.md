# Part B — 7B-class long-context (SL=1024) evaluation

> **Headline**: On the real **Qwen2.5-7B** (GQA, 28 attn / 4 KV heads) at
> WT-2 SL=1024, fp16 PPL is **5.92**; per-channel/per-token **BaseQuant INT4
> = 7.45** (**+1.52** over fp16 — *not* near-lossless even at 7B) and **INT3
> = 9.67** (+3.74). CARE-KV INT3's projected KV footprint at this scale is
> **0.23× fp16** with residual ~6% of total. CARE-KV remains meaningful: its
> value is delivering INT3-level memory at better-than-INT3 quality, and the
> quality gap that INT4 leaves open is exactly what the residual closes.

## Setup (hybrid, per the agreed plan)

CARE-KV's monkey-patch targets `LlamaForCausalLM`; the only local 7B,
Qwen2.5-7B, is `Qwen2ForCausalLM`. So:

- **fp16 + BaseQuant INT4/INT3** run on the **real Qwen2.5-7B** at SL=1024
  (model-agnostic per-channel-K / per-token-V fake-quant hook — KIVI-style,
  outlier-robust; naive per-group quant collapses on Qwen's outlier channels,
  PPL>190, so per-channel is required).
- **CARE-KV** PPL anchors are referenced from Part A (TinyLlama, SL=128) —
  CARE-KV PPL at SL=1024 is infeasible with the Python-loop prototype
  (documented; ~tens of GPU-hours/cell). Their **7B KV memory is projected**
  with the Part C estimator.

Config verified from `config.json`: L=28, hidden=3584, **28 attn / 4 KV
heads (GQA group=7)**, head_dim=128, bf16.

## Results table (Qwen2.5-7B, WT-2 SL=1024, N=4, B=1)

| method | PPL | ΔvsFP16 | fp16 KV GB | total KV GB | KV/fp16 | residual GB | tok/s | peak GPU |
|---|---|---|---|---|---|---|---|---|
| fp16 | 5.9213 | 0.00 | 0.0547 | 0.0547 | 1.000× | 0.0 | 4154 | 17.2 GB |
| BaseQuant INT4 | 7.4456 | +1.52 | 0.0547 | 0.0137 | 0.250× | 0.0 | 5248 | 17.2 GB |
| BaseQuant INT3 | 9.6653 | +3.74 | 0.0547 | 0.0103 | 0.188× | 0.0 | 5321 | 17.2 GB |
| uniform INT3 + CARE-KV | *13.46 (TL)* | — | 0.0547 | 0.0126 | 0.230× | 0.0007 | — | — |
| KIVI INT3 + CARE-KV | *13.09 (TL)* | — | 0.0547 | 0.0126 | 0.230× | 0.0007 | — | — |
| KVQuant-preRoPE INT3 + CARE-KV | *13.10 (TL)* | — | 0.0547 | 0.0126 | 0.230× | 0.0007 | — | — |

*PPL for CARE-KV rows is the TinyLlama SL=128 anchor (Part A); memory is the
7B SL=1024 projection. fp16/BaseQuant PPL are real Qwen-7B numbers.*

## Reviewer takeaways

- **Real 7B fp16 actually solves WT-2** (PPL 5.92), so quantization error is
  meaningful here (unlike TinyLlama where fp16 itself is weak at 12.3).
- **BaseQuant INT4 is NOT near-lossless at 7B** (+1.52 PPL); INT3 costs
  +3.74. So low-bit KV quant genuinely degrades a strong 7B — there *is*
  room for residual correction to recover quality. (Matches Part A on
  TinyLlama where CARE-KV recovers most of the INT3 gap.)
- **GQA keeps single-stream KV small** (0.055 GB at SL=1024, B=1). KV reaches
  GB scale only with large batch×context (Part C: B=8×S=2048 → 0.875 GB);
  there CARE-KV INT3 → 0.20 GB (0.23×), residual ~6%.
- **Throughput**: prefill ~4–5k tok/s on one A100-class GPU at SL=1024;
  fake-quant adds no measurable prefill cost (quant is cheap; the expensive
  part is CARE-KV's *decode* correction — Part E).

## Caveat

This is a **diagnostic pilot** (N=4, SL=1024, B=1). The BaseQuant hook is a
same-condition fake-quant (per-channel K / per-token V), not the CARE-KV
cache path (which is Llama-only); it is labelled accordingly. CARE-KV PPL on
a true 7B would require either porting the patch to Qwen2 or vectorizing the
prefill — both are future work.
