# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CARE-KV: a research implementation of low-bit (INT2/INT3/INT4) KV-cache
quantization for LLaMA-family models, augmented with **sparse residual
correction** chosen by an output-error-aware router. Targets HuggingFace
`LlamaForCausalLM` via monkey-patching.

## Import / path convention

Package imports as **`CARE_KV.care_kv`**, not `care_kv`. Repo lives at
`~/CARE_KV/care_kv/`. All scripts assume `PYTHONPATH=/home/soeun` (the
grandparent of this directory). Always:

```bash
cd /home/soeun/CARE_KV/care_kv
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun
```

---

## 1. Current project status — method-complete

Stable and validated:

- **Post-RoPE K storage** in the cache (decode reads RoPE-correct K).
- **GQA-aware KV-head-indexed cache layout** (no per-query-head duplication).
- `CAREKV_PREFILL_MODE=carekv_stored` reads **only real stored residual slots**
  (no oracle leak, no upper-bound shortcut).
- `use_cache=True` / HF `DynamicCache` incremental decode works end-to-end
  (open-page append, no fresh page per token).
- **Packed INT2/INT3/INT4 base KV** storage (`CAREKV_PACKED_BASE=1`).
- **`scale_quant=int8`** with per-page master scale.
- **Absolute K/V budgets** (`STORE_ABS_K/V`, `READ_ABS_K/V`) — the current
  paper path; ratio budgets are legacy.
- **Score-normalized `joint` routing** is the current best policy.
- `correction_impl=cached` (pre-unpacked-slot cache per kv_head) is the
  **main stable runtime path**.
- `correction_impl=vectorized` exists and matches cached within fp16 noise
  for V-only, but **joint+both falls back to cached for bit-exactness**.
- `R=0 / READ_BUDGET=0 ≡ base_quant` invariant is exact (pytest-locked).

---

## 2. Paper-best configuration (locked)

Do not silently change these. If you ablate, gate behind a flag and keep
the defaults below.

```
BASE_BITS=3
CAREKV_PACKED_BASE=1
CAREKV_SCALE_QUANT=int8
CAREKV_PREFILL_MODE=carekv_stored
CAREKV_PREFILL_RESIDUAL_KIND=both
CAREKV_ROUTE_POLICY=joint
CAREKV_SCORE_NORMALIZE=1
CAREKV_CORRECTION_IMPL=cached
CAREKV_BUDGET_POLICY=uniform
STORE_ABS_K=2
STORE_ABS_V=4
READ_ABS_K=2
READ_ABS_V=2
```

---

## 3. Main result summary

Synthetic sanity (SEQ_LEN=64, TinyLlama-1.1B):

- base_quant INT3 PPL **≈ 4.2831**
- optimized CARE-KV PPL **≈ 3.8294**
- improvement **≈ 10.6%**

Memory (`memory_table.csv`):

- packed_base + scale_quant=int8 reaches **≈ 0.24–0.26× FP16 KV** at
  context ≥ 2k.

Decode:

- `use_cache=True` incremental decode works.
- Latency is **prototype-level**: per-(layer, kv_head, t) Python loops in
  prefill + cached correction; HF DynamicCache dummy fp16 K/V inflates
  peak GPU memory. Both are documented in `summaries/remaining_improvements.md`.

WikiText-2 paper run (TinyLlama, SL=128, N=16): see
`results/paper_eval_20260529_015053/final_report.md` §9.

---

## 4. Historical pitfalls — do not repeat

- **Old zero-read `carekv_stored` rows are invalid.** Any row with
  `K_reads=0` AND `V_reads=0` under `carekv_stored` means the router
  never fired (typically because `STORE_BUDGET_RATIO=0` AND no absolute
  budget was set). Treat as superseded.
- **`carekv_eval` is an upper-bound diagnostic, not the paper method.**
  Never present `carekv_eval` numbers as final results.
- **The paper method is `carekv_stored`.**
- **Ratio budgets saturate.** `STORE_BUDGET_RATIO` / `READ_BUDGET_RATIO`
  hit a flat region quickly; use absolute budgets (`STORE_ABS_K/V`,
  `READ_ABS_K/V`) for the paper path.
- **`u_shaped` / `sensitivity` layer-budget policies did NOT improve over
  `uniform`** at the optimized absolute-budget level. Keep `uniform`.
- **K-only was best _only_ before absolute-budget tuning.** Current best
  is `kind=both` + `route_policy=joint` + `score_normalize=1`.
- **READ=0 invariant must hold.** `STORE_*=0` and `READ_*=0` (or
  `RESIDUAL_RATIO=0 + MIN_RESIDUALS=0`) must produce output bit-identical
  to `base_quant`. This is the invariant gating safe refactors.
- **Always check `K_reads`/`V_reads` for `carekv_stored` results.**
  Zero reads = router didn't fire = invalid CARE-KV claim.

---

## 5. Paper-eval directory structure

```
results/paper_eval_20260529_015053/
├── README.md                ← reproduce guide + headline numbers
├── final_report.md          ← auto-generated single-page summary
├── artifact_list.txt
├── summaries/               ← hand-curated markdown tables
├── figures/                 ← per-layer PNGs + 8 paper summary PNGs (fig_*.png)
├── ppl/                     ← synthetic-prompt PPL (sanity)
├── ppl_dataset/             ← WikiText-2 paper N=16 + multi-model N=4
├── sweeps/                  ← budget / Pareto sweeps
├── ablations/               ← V/K/Both, route policy, layer budget
├── memory/                  ← measured KV memory vs estimator vs fp16
├── latency/                 ← prefill + decode + vectorization bench
├── long_context/            ← Phase J synthetic retrieval/copy
├── generation/              ← qualitative samples
└── logs/                    ← gitignored
```

---

## 6. Main scripts / tools

End-to-end:

- `scripts/run_all_paper_eval.sh` — full paper-matrix runner (~4 h on one
  A100/H100). Honors env knobs `WT2_NUM_SAMPLES`, `LONG_CTX_TRIALS`,
  `SKIP_HEAVY=1` for fast smoke.

Per-section eval drivers:

- `scripts/run_wikitext2_ppl.sh`           — WT-2 PPL on the paper-best config
- `scripts/run_multimodel_ppl_eval.sh`     — WT-2 across a `$MODELS` list
- `scripts/run_long_context_retrieval.py`  — Phase J synthetic tasks
- `scripts/debug_stored_slot_reads.sh`     — sanity check the router is firing
- `tools/paper_eval.py {memory,figures,...}` — multi-subcommand driver
- `tools/eval_prefill_vectorization.py`    — Phase P4 vectorized-V bench
- `tools/eval_layer_budget_policies.py`    — Phase E layer budget ablation
- `eval_ppl_dataset.py`                    — HF-datasets PPL driver (wikitext/c4)

Wrap-up:

- `tools/summarize_all_results.py <paper_dir>` — regenerate `final_report.md`
  + `artifact_list.txt` from the CSVs.
- `tools/make_paper_figures.py --paper-dir <paper_dir>` — generate the 8
  paper summary PNGs.

---

## 7. What to run first in a new session

Safe order — confirms env, source builds, tests pass, and the existing
paper artifacts are still intact:

```bash
cd /home/soeun/CARE_KV/care_kv
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

# 1. Source compiles
python -m py_compile quantizer.py cache.py residual_store.py \
    residual_router.py attention.py layer.py llama_patch.py utils.py \
    __init__.py tools/*.py

# 2. Tests pass (18 tests, ~70 s)
python -m pytest -q tests/test_carekv_v2.py

# 3. Regenerate the report from existing CSVs (no experiment re-run, ~1 s)
python tools/summarize_all_results.py results/paper_eval_20260529_015053
```

For a fast smoke of the full pipeline (skip heavy carekv cells, ~15 min):
```bash
PAPER_DIR=results/paper_eval_20260529_015053 SKIP_HEAVY=1 \
  bash scripts/run_all_paper_eval.sh
```

---

## 8. Remaining work

Engineering:

- More robust WikiText-2 / C4 paper-scale runs (SL ≥ 512, N ≥ 32) once
  joint+both is vectorized.
- Multi-model coverage beyond TinyLlama + JackFram/llama-160m (needs HF
  auth or larger local cache).
- Stronger long-context benchmark (RULER / LongBench) or a larger model
  where fp16 actually solves the task at scale.
- **Vectorized joint+both correction** — top-priority unblocker for
  large-SL paper evals.
- **Custom lightweight HF Cache** to avoid the DynamicCache dummy fp16
  K/V we currently feed `get_seq_length()` (would cut peak GPU mem ~50%).
- CUDA / Triton kernels for packed unpack + correction.

Paper:

- Final paper writing.

---

## 9. Rules for future work

1. **Do not silently change the paper-best configuration** (§2).
2. **Keep old behavior behind flags.** Don't rip out `carekv_eval`,
   ratio-budget paths, or non-packed base — they're still useful for
   ablation and would break old result reproduction if removed.
3. **Label results clearly** in every report:
   - `sanity / smoke` — short synthetic prompt, N ≤ 4.
   - `diagnostic` — per-layer or debugging artifact.
   - `prototype latency` — current Python-loop wall-clock, not the
     achievable runtime.
   - `paper-ready` — passes the invariant + config gates and is in
     `final_report.md`.
4. **Never present `carekv_eval` as the final method.** Paper method is
   `carekv_stored`.
5. **Never present zero-read `carekv_stored` rows as final.** Always
   check `K_reads + V_reads > 0` for any CARE-KV claim.
6. **Always check `K_reads` / `V_reads`** for `carekv_stored` cells
   before believing the PPL.
7. **Preserve the `READ=0 ≡ base_quant` invariant.** It is the gate for
   safe refactors of the router and correction paths.
8. **Preserve both `USE_CACHE=0` and `USE_CACHE=1` paths.** The
   eval-script path uses `USE_CACHE=0` (full prefill); the streaming
   generation path uses `USE_CACHE=1` (DynamicCache + open-page append).
   Don't conflate or remove either.

---

## Runtime knobs cheat-sheet

| Env var | Values | Effect |
|---|---|---|
| `CAREKV_PREFILL_MODE` | `fp` / `base_quant` / `carekv_eval` / `carekv_stored` | Prefill path. **Use `carekv_stored` for paper.** |
| `CAREKV_PREFILL_RESIDUAL_KIND` | `v` / `k` / `both` | Which residual to apply. **`both` for paper.** |
| `CAREKV_ROUTE_POLICY` | `joint` / `separate` / `k_first` / `adaptive` | Routing policy. **`joint` for paper.** |
| `CAREKV_SCORE_NORMALIZE` | 0 / 1 | Per-kind normalization for `joint`. **1 for paper.** |
| `CAREKV_CORRECTION_IMPL` | `python` / `cached` / `vectorized` | Correction kernel. **`cached` for paper.** |
| `CAREKV_BUDGET_POLICY` | `uniform` / `u_shaped` / `sensitivity` | Per-layer budget multiplier. **`uniform` for paper.** |
| `CAREKV_PACKED_BASE` | 0 / 1 | Real packed INT base storage. **1 for paper.** |
| `CAREKV_SCALE_QUANT` | `none` / `int8` | Per-page scale quantization. **`int8` for paper.** |
| `BASE_BITS` | 2 / 3 / 4 | Base KV bit-width. **3 for paper.** |
| `STORE_ABS_K`, `STORE_ABS_V` | int | Absolute per-page store budget (K, V). **2, 4 for paper.** |
| `READ_ABS_K`, `READ_ABS_V` | int | Absolute per-decode read budget (K, V). **2, 2 for paper.** |
| `CAREKV_DEBUG_STATS` | 0 / 1 | Emit `K_reads` / `V_reads` counters. **Always set 1 for CARE-KV runs** — needed to validate the router fired. |
| `MODEL_ID` | HF id | Defaults to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. |
| `SEQ_LEN`, `NUM_SAMPLES` | int | PPL window length / count. |
