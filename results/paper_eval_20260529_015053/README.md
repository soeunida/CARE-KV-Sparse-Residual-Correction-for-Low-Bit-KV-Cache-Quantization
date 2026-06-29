# CARE-KV paper-evaluation artifact bundle

This directory is the single source of truth for the CARE-KV paper-quality
evaluation. Every CSV, summary, figure, and report referenced by the paper
draft lives here.

## How to read this directory

- **`final_report.md`** — auto-generated single-page summary of every result
  (PPL tables, ablations, multi-model, long-context, figures). Start here.
- **`README.md`** — this file (reproduce guide).
- **`artifact_list.txt`** — flat list of every committed artifact.
- **`summaries/`** — hand-curated markdown tables for each section; the
  `final_report.md` numbers come from the CSVs, the narrative comes from
  these.
- **`figures/`** — diagnostic PNGs (per-layer) + 8 paper summary PNGs
  (`fig_*.png`) generated from the CSVs.
- **`ppl/`** — short synthetic-prompt PPL (sanity, not paper headline).
- **`ppl_dataset/`** — WikiText-2 paper run (N=16) + multi-model smoke (N=4).
- **`sweeps/`** — budget / Pareto sweeps (stored ratio, absolute budget,
  memory–PPL Pareto).
- **`ablations/`** — V/K/Both, route policy, layer budget policy.
- **`memory/`** — measured KV-cache memory vs estimator vs fp16.
- **`latency/`** — prefill + decode latency, prefill vectorization bench.
- **`long_context/`** — Phase J synthetic retrieval/copy results.
- **`generation/`** — qualitative generation samples (fp16 / base_quant / CARE-KV).
- **`logs/`** — per-step build logs (gitignored).

## Paper-best CARE-KV config (locked)

```
CAREKV_PREFILL_MODE=carekv_stored
CAREKV_PREFILL_RESIDUAL_KIND=both
CAREKV_ROUTE_POLICY=joint
CAREKV_SCORE_NORMALIZE=1
CAREKV_CORRECTION_IMPL=cached         # vectorized falls back to cached for joint+both
CAREKV_BUDGET_POLICY=uniform
CAREKV_PACKED_BASE=1
CAREKV_SCALE_QUANT=int8
STORE_ABS_K=2  STORE_ABS_V=4
READ_ABS_K=2   READ_ABS_V=2
BASE_BITS=3
```

## Headline numbers (TinyLlama-1.1B-Chat-v1.0, WikiText-2, SL=128, N=16)

| mode | PPL | Δ vs fp16 |
|---|---:|---:|
| fp16                       | 15.77 | +0.00 |
| INT4 base_quant            | 16.44 | +0.67 |
| **CARE-KV INT3 (paper)**   | **18.14** | **+2.37** |
| INT3 base_quant            | 21.74 | +5.97 |
| INT2 base_quant            | 351.60 | +335.83 |

CARE-KV INT3 closes **(21.74 − 18.14) / (21.74 − 15.77) ≈ 60 %** of the
INT3 → fp16 PPL gap, while keeping INT3 memory.

## End-to-end reproduce

From the repo root (`~/CARE_KV/care_kv/`):

```bash
# 1. Activate env + set PYTHONPATH
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

# 2. Run everything into this directory
PAPER_DIR=results/paper_eval_20260529_015053 \
  bash scripts/run_all_paper_eval.sh
```

Compute envelope on one A100/H100-class GPU: **~4 h** total (WikiText-2
dominates ~1.8 h, long-context dominates ~1.7 h, everything else is minutes).

Drop heavy cells for a fast smoke run:
```bash
PAPER_DIR=... WT2_NUM_SAMPLES=4 LONG_CTX_TRIALS=2 \
  bash scripts/run_all_paper_eval.sh        # ~15 min
```

## Per-section reproduce

| Section in `final_report.md` | Script |
|---|---|
| §3 memory | `python tools/paper_eval.py memory ...` |
| §4 absolute budget sweep | `bash scripts/run_paper_eval_clean.sh` (cell D) |
| §5 routing policy ablation | `python tools/eval_route_policies.py ...` |
| §6 layer budget policy | `python tools/eval_layer_budget_policies.py ...` |
| §7 use_cache=True latency | `bash scripts/bench_latency.sh` |
| §8 prefill vectorization | `python tools/eval_prefill_vectorization.py ...` |
| §9 WikiText-2 paper PPL | `bash scripts/run_wikitext2_ppl.sh` (NUM_SAMPLES=16) |
| §10 multi-model | `bash scripts/run_multimodel_ppl_eval.sh` |
| §11 long-context | `python scripts/run_long_context_retrieval.py ...` |
| §12 layer diagnostics | `python tools/paper_eval.py figures ...` |
| §12 paper summary figures | `python tools/make_paper_figures.py --paper-dir <dir>` |

## Known limitations (not a CARE-KV claim)

- **`copy` task** (Phase J): TinyLlama fp16 itself scores EM=0 — excluded
  as a CARE-KV result. The template needs rework for small models.
- **`boundary` task** (Phase J): CARE-KV regresses vs base_quant in the
  close-distractor regime — flagged for a future routing-policy ablation
  rather than the headline.
- **Prefill latency** is O(T² × H_q × L) Python loop. Vectorized V exists
  but joint+both still uses cached for bit-exactness. Blocks SL ≥ 512
  paper evals until joint+both is vectorized.
- **No multi-model coverage beyond TinyLlama + JackFram/llama-160m** in this
  bundle. Llama-3.2-1B is HF-gated; larger models exceed session disk budget.

See `summaries/remaining_improvements.md` for the prioritized list of
follow-up engineering items.
