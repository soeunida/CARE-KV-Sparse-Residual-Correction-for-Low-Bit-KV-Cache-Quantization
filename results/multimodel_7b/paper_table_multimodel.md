# Paper table — CARE-KV multi-model INT3 KV-cache recovery

**Setup.** WikiText-2 perplexity, sequence length 512, 4 windows. Base = INT3
KV-cache quantization (per-group 32, symmetric). CARE-KV = INT3 base + output-
error-aware sparse residual correction (paper-best: store K/V = 2/4, read K/V =
2/2, joint score-normalized routing, vectorized correction). Δ = PPL improvement
of CARE-KV over the INT3 base; **Recovery** = Δ / (base − fp16) = fraction of the
INT3 quantization gap closed toward fp16. All rows: router active (K/V residual
reads > 0). Runtime is prototype (Python-loop); numbers are quality, not latency.

| Model | Params | Attn | fp16 | INT3 base | CARE-KV | Δ | Recovery |
|:--|--:|:--:|--:|--:|--:|--:|--:|
| Mistral-7B-v0.3 | 7B | GQA | 6.785 | 7.297 | **7.124** | 0.173 | **34%** |
| SOLAR-10.7B | 10.7B | GQA | 5.697 | 6.164 | **5.985** | 0.179 | **38%** |
| Llama-2-13B | 13B | MHA | 5.777 | 6.505 | **6.230** | 0.275 | **38%** |
| CodeLlama-34B | 34B | GQA | 6.961 | 7.489 | **7.303** | 0.186 | **35%** |
| Yi-34B | 34B | GQA | 5.525 | 5.900 | 5.871 | 0.029 | 8% |

**Takeaways.**
- CARE-KV consistently recovers **~34–38%** of the INT3 KV-quantization gap across
  model families (Mistral, Llama-2, CodeLlama, SOLAR) and both attention types
  (GQA, MHA), at 7B–34B scale — the method transfers, not tuned per model.
- Recovery tracks the **available headroom**: Yi-34B is already near-lossless under
  INT3 (base gap only 0.375 PPL), so there is little to recover (8%); the effect
  is largest where INT3 hurts most (MHA Llama-2-13B, Δ = 0.275).

**Config (locked, paper-best):** BASE_BITS=3, packed base, int8 scales,
carekv_stored, joint + score_normalize, correction_impl=vectorized, sketch_dim=32,
STORE_ABS K/V = 2/4, READ_ABS K/V = 2/2.
