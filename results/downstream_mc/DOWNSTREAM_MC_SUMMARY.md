# CARE-KV downstream multiple-choice evaluation (DeepSeek-7B)

Reviewer ask: evaluate CARE-KV on ≥1–2 downstream tasks beyond WikiText-2 PPL.
Method: log-probability multiple-choice scoring (single `use_cache=False` forward
per choice — the same path the PPL number uses, so the quantized KV genuinely
participates). Model: `deepseek-ai/deepseek-llm-7b-base`, 0-shot, seed 0.
Tool: `tools/eval_downstream_mc.py`.

## A. Robust comparison at n=500 (fast modes: fp16 vs turboquant INT3)

Both tasks, full n=500 (binomial 95% CI ≈ ±4.3 pts).

| mode | MMLU acc | ARC acc | ARC acc_norm | n |
|---|---:|---:|---:|---:|
| fp16 (reference) | **0.400** | **0.446** | **0.434** | 500 |
| turboquant INT3 (QJL) | 0.374 | 0.428 | 0.426 | 500 |
| Δ (turbo − fp16) | −0.026 | −0.018 | −0.008 | |

turboquant INT3 is a small, consistent drop below fp16 on both tasks at n=500 —
within/near the CI but consistent in sign. (Note the earlier n=64 turbo MMLU of
0.4062 was small-sample noise; the n=500 value is 0.374.)

## B. Slow modes at n=64 (base_quant / CARE-KV — prototype-limited)

MMLU only, matched n=64 (same 64 questions). 95% CI ≈ ±12 pts — indicative only.

| mode | MMLU acc (n=64) | correct | K_reads | V_reads | valid | s/example |
|---|---:|---:|---:|---:|:--:|---:|
| fp16 (reference) | 0.4531 | 29/64 | 0 | 0 | — | 0.07 |
| base_quant INT3 | 0.4062 | 26/64 | 0 | 0 | — | 32.0 |
| turboquant INT3 | 0.4062 | 26/64 | 0 | 0 | — | 0.14 |
| **CARE-KV INT3 (paper)** | 0.4062 | 26/64 | 10,800,731 | 14,224,549 | **yes** | 114.5 |

CARE-KV validated (router fired, K_reads+V_reads ≫ 0 — not a zero-read artifact,
per CLAUDE.md). At n=64 all three INT3 methods tie (26/64); differences vs fp16
and among methods are within the ±12 pt noise floor.

Why base_quant / CARE-KV stay at n=64 and skip ARC: base_quant ≈ 32 s/ex and
CARE-KV ≈ 115 s/ex through the audited adapter prefill, so MMLU n=500 is 6–20 h and
ARC (≈4 forwards/example) is 6–20 h *more* per mode — the documented prototype
runtime blocker. turboquant is fast (~0.14 s/ex) and is therefore reported at full
n=500 (§A). A larger-n CARE-KV / any CARE-KV ARC needs the vectorized correction
path (CLAUDE.md §8).

## Honest reading

- **Two downstream tasks beyond WikiText-2 are covered** (MMLU + ARC-Challenge),
  with CARE-KV validated (router fired).
- **At n=500 (fp16 vs turboquant):** INT3 KV (turboquant) costs a small, consistent
  −0.9 to −2.6 pts vs fp16 across MMLU/ARC. This is the robust part of the story.
- **At n=64 (base_quant / CARE-KV):** INT3 KV is close to fp16 within a wide noise
  floor, and **CARE-KV neither helps nor hurts multiple-choice argmax vs
  base_quant** at this budget. Its value stays its PPL / KV-reconstruction result,
  not MC accuracy.
- **No downstream *win* is claimed for CARE-KV.** No speedup is claimed; latencies
  above are the Python-loop prototype, not achievable runtime.

Data: `results/downstream_mc/deepseek_{fp16,fp16_n64,baseq,turbo,turbo_n500,carekv}.csv`.
