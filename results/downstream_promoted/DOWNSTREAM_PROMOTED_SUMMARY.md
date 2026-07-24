# Downstream MC (MMLU / ARC / LAMBADA) — promoted config, real

Forward-pass (logprob) task-level evaluation — the **feasible** task benchmark
for CARE-KV (unlike LongBench/RULER generation, which is decode-kernel-blocked).
Config = promoted paper-best (`exact` K correction + `combined` selector,
`vectorized`). WikiText-2-style windowing; `use_cache=False` single forward per
example, so CARE-KV runs at PPL-regime cost, not autoregressive-generation cost.

**N=100 per task, seed 0.** Router fired on every CARE-KV cell (K/V reads > 0).

| model | task | fp16 | base INT3 | Turbo INT3 | CARE-KV | Δ vs base | Δ vs Turbo |
|---|---|---:|---:|---:|---:|---:|---:|
| Mistral-7B | MMLU | 0.56 | 0.56 | 0.51 | 0.52 | −0.04 | +0.01 |
| Mistral-7B | ARC (acc) | 0.53 | 0.51 | 0.52 | 0.51 | +0.00 | −0.01 |
| Mistral-7B | ARC (norm) | 0.55 | 0.53 | 0.48 | 0.54 | +0.01 | +0.06 |
| Mistral-7B | LAMBADA | 0.85 | 0.81 | 0.85 | 0.86 | +0.05 | +0.01 |
| DeepSeek-7B | MMLU | 0.43 | 0.36 | 0.39 | 0.41 | +0.05 | +0.02 |
| DeepSeek-7B | ARC (acc) | 0.47 | 0.39 | 0.42 | 0.49 | +0.10 | +0.07 |
| DeepSeek-7B | ARC (norm) | 0.47 | 0.40 | 0.48 | 0.48 | +0.08 | +0.00 |
| DeepSeek-7B | LAMBADA | 0.83 | 0.77 | 0.80 | 0.79 | +0.02 | −0.01 |

## Honest reading

- **The gain tracks how much INT3 actually damages the task.** On the
  outlier-heavy model (DeepSeek) INT3 drops MMLU 0.43→0.36 and ARC 0.47→0.39;
  CARE-KV recovers most of it (MMLU 0.41, ARC 0.49) and **beats base on all 4
  tasks, beats/ties Turbo on 3/4** (ARC-acc is a clear +0.07 over Turbo). On the
  mild-outlier model (Mistral) INT3 barely moves MMLU/ARC (fp16≈base), so there
  is little to recover — CARE-KV mostly ties, with a small LAMBADA win.
- This **matches the paper's own thesis** ("gains concentrate where
  quantization error is large") with real, forward-pass evidence, and is the
  honest replacement for the unbacked LongBench/RULER Table 2.
- **CARE-KV recovers INT3 downstream degradation; on short-context MC it does
  not manufacture a gap that INT3 didn't open.**

## Caveats (do not overstate)

- **N=100** → ~±0.05–0.10 CI. DeepSeek ARC-acc +0.10 is at the edge of
  significance; MMLU +0.05 is borderline. **Larger N (≥300–500) is required
  before any of these go in the paper.** The *direction* (CARE-KV > base on the
  damaged tasks, competitive with Turbo) is consistent across MMLU+ARC on
  DeepSeek, which strengthens it.
- **2 models** (Mistral mild-outlier, DeepSeek outlier-heavy). More primary
  models needed for a full table.
- MC tasks are short-context, so they under-represent CARE-KV's long-context
  advantage (which needs the decode kernel to measure — LongBench/RULER remain
  blocked; see CLAUDE.md §5f and the eval_longbench_real.py 211s+/sample wall).

Data: `results/downstream_promoted/{mistral7b,deepseek7b}.csv`.
Driver: `tools/eval_downstream_mc.py` with `CAREKV_K_CORRECTION_MODE=exact
CAREKV_KSCORE_LIVE=1`.
