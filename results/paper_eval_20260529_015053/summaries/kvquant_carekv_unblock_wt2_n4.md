# KVQuant-style (pre-RoPE) + CARE-KV unblock — WikiText-2 N=4 SL=128

**Label:** `screening` (WT-2 N=4, SL=128, TinyLlama-1.1B). Small sample — the
sub-0.5-PPL differences between the CARE-KV stacks need N≥16 to be conclusive.
All CARE-KV cells fired the router (`K_reads + V_reads > 0`), so the corrections
are valid (not zero-read).

KVQuant-style here is a **same-condition reimplementation** (pre-RoPE per-channel
K quantization), NOT official KVQuant. The "unblock" is that CARE-KV's residual,
which is computed in post-RoPE coordinates, can now be stacked on a pre-RoPE
KVQuant base (K is quantized pre-RoPE, K_hat re-rotated, residual taken in
post-RoPE space). Previously this cell was recorded `unsupported`.

- model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, BASE_BITS=3, CARE-KV paper-best
  store/read (SK2 SV4 RK2 RV2), `carekv_stored`, max_pages=16.
- GPU note: run executed on an idle GPU (the default GPU 0 was contended by an
  unrelated 41 GB job; one transient OOM on that GPU was avoided by relocation,
  not a method failure).

## Result table

| method | family | PPL | ΔvsFP16 | ΔvsINT3 | est. KV MB | vs fp16 | K_reads | V_reads | rt(s) |
|:-------|:-------|----:|------:|------:|------:|------:|------:|------:|----:|
| fp16 | fp16 | 12.3457 | 0.0000 | −3.8516 | 2.75 | 1.000× | 0 | 0 | 24 |
| base_quant_INT3 | base_quant | 16.1973 | +3.8516 | 0.0000 | 0.52 | 0.188× | 0 | 0 | 30 |
| KVQuant_style_INT3 pre-RoPE (standalone) | kvquant_style | 15.0080 | +2.6623 | −1.1893 | 0.55 | 0.199× | 0 | 0 | 17 |
| **KVQuant pre-RoPE + CARE-KV** | kvquant_plus_carekv | **13.1004** | +0.7547 | −3.0969 | 0.69 | 0.249× | 648597 | 793195 | 2093 |
| KIVI INT3 + CARE-KV | kivi_plus_carekv | 13.0948 | +0.7491 | −3.1025 | 0.69 | 0.249× | 648980 | 792812 | 1997 |
| uniform INT3 + CARE-KV (paper-best) | care_kv | 13.4618 | +1.1161 | −2.7355 | 0.65 | 0.238× | 641915 | 799877 | 1669 |

## Findings

**1. Does KVQuant-style pre-RoPE + CARE-KV improve over KVQuant-style standalone?**
**YES, substantially.** 15.0080 → **13.1004** = **−1.9076 PPL** (−12.7% relative).
Stacking CARE-KV residual correction on the pre-RoPE KVQuant base recovers most of
the remaining gap to fp16 (the stack is +0.75 from fp16, vs +2.66 standalone). The
unblock is validated: the cell is now a working `same-condition reimplementation`,
no longer `unsupported`.

**2. Does it beat KIVI-style INT3 + CARE-KV?**
**No — they are tied within noise.** KVQuant+CARE-KV 13.1004 vs KIVI+CARE-KV
13.0948 → KIVI is ahead by just **0.0056 PPL** (0.04%), far inside fp16 rounding
noise at N=4. Practically equivalent. Both pre-RoPE/KIVI bases + CARE-KV are
~0.36 PPL better than the uniform paper-best stack here, but that gap also needs
N≥16 to confirm.

**3. Memory.** All three CARE-KV stacks land at ~0.65–0.69 MB estimated KV
(~0.24–0.25× fp16). KVQuant+CARE and KIVI+CARE are identical at 0.6854 MB
(0.249×); the uniform paper-best stack is slightly leaner at 0.6531 MB (0.238×)
because uniform per-group INT3-packed base is marginally smaller than the
per-channel-K / per-token-V bases (which carry fp16 scale headers). The memory
cost of the KVQuant unblock over the paper-best is therefore small (+0.03 MB,
~+1% of fp16).

**4. Router activity (K_reads / V_reads).** All CARE-KV cells fired strongly and
comparably: KVQuant+CARE 648597 / 793195, KIVI+CARE 648980 / 792812, uniform+CARE
641915 / 799877. No zero-read rows → all CARE-KV PPLs are valid.

**5. Runtime.** Cheap cells < 30 s; the three `carekv_stored` cells are the
prototype Python-loop correction path: 2093 s (KVQuant+CARE), 1997 s (KIVI+CARE),
1669 s (uniform+CARE). Prototype latency, not achievable runtime.

## Conclusion

The KVQuant-style pre-RoPE + CARE-KV stack is **unblocked and validated**: it works
end-to-end, fires the router, and improves massively over KVQuant standalone
(−1.91 PPL). It is **statistically tied with KIVI + CARE-KV** (Δ=0.006 PPL) and, on
this N=4 screen, both edge out the uniform paper-best stack by ~0.36 PPL at ~equal
memory. These sub-0.5-PPL orderings are within N=4 noise; **N≥16 confirmation is
required** before treating KVQuant/KIVI+CARE-KV as superior to the uniform
paper-best. The paper-best configuration is unchanged pending that confirmation.
