# CARE-KV downstream multiple-choice evaluation (DeepSeek-7B)

Reviewer ask: evaluate CARE-KV on ≥1–2 downstream tasks beyond WikiText-2 PPL.
Method: log-probability multiple-choice scoring (single `use_cache=False` forward
per choice — the same path the PPL number uses, so the quantized KV genuinely
participates). Model: `deepseek-ai/deepseek-llm-7b-base`, 0-shot, seed 0.
Tool: `tools/eval_downstream_mc.py`.

## MMLU (matched n=64, same questions across all modes)

| mode | MMLU acc (n=64) | correct | K_reads | V_reads | valid | s/example |
|---|---:|---:|---:|---:|:--:|---:|
| fp16 (reference) | **0.4531** | 29/64 | 0 | 0 | — | 0.07 |
| base_quant INT3 | 0.4062 | 26/64 | 0 | 0 | — | 32.0 |
| turboquant INT3 (QJL) | 0.4062 | 26/64 | 0 | 0 | — | 0.14 |
| **CARE-KV INT3 (paper)** | 0.4062 | 26/64 | 10,800,731 | 14,224,549 | **yes** | 114.5 |

fp16 reference at full n=500: MMLU acc **0.400**.

## ARC-Challenge (fp16 only, n=500)

| mode | acc | acc_norm | n |
|---|---:|---:|---:|
| fp16 | 0.446 | 0.434 | 500 |

Quantized-mode ARC not run: each example needs one forward **per choice** (~4×),
which at 32–115 s/example (base_quant / CARE-KV) is 6–20 h/mode — the documented
prototype-prefill runtime blocker.

## Honest reading

- **CARE-KV is valid here** — the router fired (K_reads+V_reads ≫ 0), so this is a
  real `carekv_stored` result, not a zero-read artifact (CLAUDE.md rule).
- **On MMLU, all three INT3 methods score identically (26/64 = 0.4062)** and sit
  ~3 questions (4.7 pts) below fp16 (29/64). CARE-KV's residual correction does
  **not** measurably improve MMLU letter-argmax over plain base_quant at this
  sample size — its gains are in PPL / KV-reconstruction fidelity, not necessarily
  in short-prompt multiple-choice argmax, which is robust to the small INT3
  perturbation.
- **These differences are within the n=64 noise floor.** A 64-example binomial CI
  is roughly ±12 pts at 95%, and fp16 itself moves 0.400 (n=500) ↔ 0.4531 (n=64)
  on sample alone. So "fp16 > INT3 by 3 questions" and "the three INT3 methods
  tie" are both **not statistically distinguishable** here; treat as indicative,
  not conclusive.
- **Why n=64 for the quantized modes**: base_quant ≈ 32 s/ex and CARE-KV ≈ 115 s/ex
  through the audited adapter prefill; n=500 would be 6 h (base_quant) to 20 h
  (CARE-KV). turboquant is fast (~0.14 s/ex, rotation-based) and could be run at
  n=500 cheaply. A larger-n CARE-KV MMLU / any CARE-KV ARC needs the vectorized /
  faster correction path (CLAUDE.md §8).

## Bottom line

A downstream task beyond WikiText-2 is now covered: **MMLU** (matched n=64 across
fp16 / base_quant / turboquant / CARE-KV, CARE-KV validated) plus **ARC-Challenge**
(fp16 n=500). On MMLU multiple-choice, INT3 KV quantization — with or without
CARE-KV — is close to fp16 (within the small-sample noise), and CARE-KV neither
helps nor hurts the letter-argmax relative to base_quant at this budget. No
speedup or downstream *win* is claimed; the value of CARE-KV remains its
PPL / KV-fidelity result, not MC accuracy.

Data: `results/downstream_mc/deepseek_{fp16,fp16_n64,baseq,turbo,carekv}.csv`.
