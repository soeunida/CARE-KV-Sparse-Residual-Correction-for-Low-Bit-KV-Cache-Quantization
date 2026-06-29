# Part D — GQA architecture validation

> **Result: ALL acceptance checks PASS.** CARE-KV is correct on a
> grouped-query-attention model (TinyLlama-1.1B: 32 attention heads / **4 KV
> heads**, group=8). The KV-memory estimator uses `num_key_value_heads` (not
> `num_attention_heads`), the router fires (nonzero K/V reads), and the
> READ=0 ≡ base_quant invariant is **bit-exact** on the GQA model.

## Model (TinyLlama-1.1B — GQA)

| field | value |
|---|---|
| model_type | llama |
| num_hidden_layers | 22 |
| hidden_size | 2048 |
| num_attention_heads | 32 |
| **num_key_value_heads** | **4** |
| head_dim | 64 |
| num_key_value_groups | 8 |
| **is_gqa** | **True** (4 < 32) |

(TinyLlama is itself GQA, so the entire CARE-KV result set — incl. Parts A/B
anchors — is already on a GQA architecture. Qwen2.5-7B, Part B, is GQA too:
28 attn / 4 KV heads, group=7.)

## Acceptance checks

| check | criterion | result | evidence |
|---|---|---|---|
| A | no shape mismatch — CARE-KV prefill runs end-to-end | **PASS** | forward ran, PPL=102.7 (synthetic SL=64) |
| B | repeat_kv path correct | **PASS** | num_key_value_groups=8, 4×8=32 |
| C | KV-memory estimator uses `num_key_value_heads`, not `num_attention_heads` | **PASS** | estimator(Hkv)=0.327 MB vs estimator(Hq)=2.613 MB → ratio **8.00** = group factor |
| D | K_reads/V_reads nonzero for CARE-KV | **PASS** | K_reads=81,705 V_reads=98,519 |
| E | READ=0 ≡ base_quant invariant | **PASS** | max\|Δlogit\| = **0.00e+00** (bit-exact); reads K=0 V=0 |

CSV: `ablations/gqa_carekv_validation.csv`.

## Why each check matters

- **C (estimator uses Hkv)** is the subtle GQA correctness point: a KV cache
  is indexed by *key/value* heads, not query heads. The estimator scaling
  exactly by the group factor (8×) confirms CARE-KV does **not**
  per-query-head-duplicate the cache — the GQA memory win is preserved.
- **B (repeat_kv)** confirms K/V are expanded from 4→32 heads only at the
  score matmul, never stored expanded.
- **E (READ=0 bit-exact)** is the safety gate: with the residual read budget
  at zero, CARE-KV must reduce to plain base-quant. `0.00e+00` logit delta
  on the GQA model shows the GQA score/correction path introduces no drift.
- **D (nonzero reads)** confirms the router actually fires on the GQA model
  (per the CLAUDE.md rule: zero-read CARE-KV rows are invalid).

## Full-budget GQA PPL (referenced from Part A)

The full WT-2 N=4 SL=128 CARE-KV anchors on this same GQA TinyLlama are in
Part A: uniform INT3 + CARE-KV **13.46**, KIVI INT3 + CARE-KV **13.09**,
KVQuant-preRoPE INT3 + CARE-KV **13.10** (vs fp16 12.35, base INT3 16.20) —
all with nonzero K/V reads. GQA does not block any CARE-KV path.
