# Long-context retrieval (Phase J) — completed at TinyLlama-tractable scale

After the first attempt failed (TinyLlama fp16 EM=0 at `n_pairs >= 20`,
`ctx_target = 512`), we re-tuned the task difficulty until fp16 actually
solves it, then ran the CARE-KV vs base_quant comparison.

## Final config

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `SEQ_LEN = ctx_target = 128`, `max_new = 6`, `num_trials = 5`
- `num_pairs = 6` (down from auto-scaled `max(20, ctx_target//12)`)
- Keys: 4-char alphanumeric (original hard variant); values: 6-char alphanumeric
- CARE-KV at paper-best config: joint+normalize+cached, packed_base=1,
  scale_quant=int8, abs SK=2 SV=4 RK=2 RV=2, uniform, K_correction_scale=0.05

## Results (`long_context/long_context_retrieval.csv`)

| task | mode | EM | char_acc | seconds | K_reads | V_reads |
|---|---|---:|---:|---:|---:|---:|
| kv_retrieval | fp16                  | **1.00** | 1.00 | 0.95 | 0 | 0 |
| kv_retrieval | base_quant_int3       | 0.00 | 0.13 | 132.6 | 0 | 0 |
| kv_retrieval | carekv_stored_int3    | 0.00 | **0.30** | 1 842.2 | 688 986 | 907 686 |
| boundary     | fp16                  | **1.00** | 1.00 | 0.55 | 0 | 0 |
| boundary     | base_quant_int3       | 0.20 | 0.33 | 132.6 | 0 | 0 |
| boundary     | carekv_stored_int3    | 0.00 | 0.07 | 1 948.6 | 709 372 | 923 908 |
| copy         | fp16                  | 0.00 | 0.00 | 0.57 | 0 | 0 |
| copy         | base_quant_int3       | 0.00 | 0.00 | 146.7 | 0 | 0 |
| copy         | carekv_stored_int3    | 0.00 | 0.00 | 2 214.6 | 749 699 | 1 052 541 |

## What this tells us

**Signal:**
- **`kv_retrieval`** — fp16 nails the 6-char target every time. INT3 base_quant
  gets 0/5 EM and ~13 % characters right (essentially random). CARE-KV INT3
  also gets 0/5 EM, but **2.3× the partial-credit** (`char_acc` 0.30 vs 0.13).
  Inspecting samples: CARE-KV reliably reproduces the first 1–3 characters of
  the target (`Q…` → `QG8W65` vs target `QD9AFZ`; `R25…` → `R25JJJ` vs target
  `R25WFU`), where base_quant produces almost-random tokens. The K/V routing
  is firing (689 k K reads, 908 k V reads across the 5 trials).

**Regression:**
- **`boundary`** — Same setup but with close-distractor keys placed around the
  target. base_quant_int3 surprisingly recovers 1/5 EM (33 % char_acc),
  while CARE-KV INT3 drops to 0/5 EM and 7 % char_acc. This is the case
  where K-residual correction was *supposed* to help most. Hypothesis: the
  routing's joint score is being dominated by the high-similarity distractor
  K vectors, so the budget is spent correcting the wrong tokens. Worth a
  follow-up ablation (separate-budget routing for `boundary`).

**Excluded:**
- **`copy`** — fp16 itself scores EM=0 on the 6-char secret-token copy at
  ctx=128, so this row carries no CARE-KV signal. The model echoes filler
  text instead of recalling the secret. Either the prompt template doesn't
  cue copying for TinyLlama-1.1B, or `max_new=6` is cut off before the
  secret position. Documented as **not a CARE-KV result** for this run.

## Paper-labelling

- `kv_retrieval` and `boundary` are reportable as a small long-context
  ablation table (partial-credit metric only — neither INT3 method achieves
  EM > 0 on a task fp16 nails, and that is itself the headline).
- `copy` row remains deferred (fp16 fails).
- A full long-context evaluation (larger model + RULER/LongBench) is still
  the right paper-final replacement; this is a CARE-KV-isolating
  microbenchmark, not a long-context claim per se.

## Files

```
long_context/long_context_retrieval.csv   # raw 9 rows + sample answers
long_context/run.log                       # full per-cell stdout
```
