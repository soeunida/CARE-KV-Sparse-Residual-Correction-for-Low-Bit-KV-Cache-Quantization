# CARE-KV: Cost-Aware Residual Error Allocation for Low-Bit KV Cache Quantization


Sparse residual correction on top of low-bit KV-cache quantization for
LLaMA-family models. Implemented as a research prototype against
HuggingFace `LlamaForCausalLM` via monkey-patching.

## What this is

At low bit-widths (INT2/INT3/INT4) base KV-cache quantization loses
quality. CARE-KV adds a sparse, query-aware **residual correction**:

```
K_hat = base_quant(K) ;  R_K = K - K_hat        # small per-page residual
V_hat = base_quant(V) ;  R_V = V - V_hat
O_care = base_attention(Q, K_hat, V_hat) + ΔO_K + ΔO_V
```

The router selects a tiny fraction of (K, V) coordinates to store
residuals for, and at decode time reads back only the top-scoring slots
to compute ΔO_K, ΔO_V. The V correction is a weighted residual sum; the
**K correction renormalizes the base softmax exactly** under the recovered
key perturbation δs = (q·R_K)/√d (bounded by construction), rather than a
1st-order Jacobian that diverges on outlier-heavy keys. The base quantizer
is pluggable — CARE-KV runs on top of its own per-group quant **or** a
KIVI-style (per-channel K, per-token V) reimplementation.

## Headline result

**WikiText-2, SL=512, NS=64, INT3 KV — CARE-KV beats TurboQuant (QJL
rotation) on all 7 tested LLaMA-architecture models.** Same `run_one`
harness / windowing for every arm; `comb+ex` is the paper-best config
(`combined_kvscore` selector + exact softmax K correction). Sorted by
K-outlier severity (`base − turbo`).

| model | fp16 | base INT3 | TurboQuant | **CARE-KV** | Δ vs Turbo |
|---|---:|---:|---:|---:|---:|
| Mistral-7B-v0.3 | 7.156 | 7.687 | 7.586 | **7.321** | **−0.265** |
| SOLAR-10.7B | 6.451 | 6.833 | 6.730 | **6.557** | **−0.173** |
| OpenLLaMA-7B-v2 | 8.603 | 9.384 | 9.108 | **8.786** | **−0.321** |
| Llama-2-13B | 6.532 | 7.279 | 6.808 | **6.731** | **−0.077** |
| Yi-6B | 7.677 | 8.826 | 8.293 | **7.931** | **−0.362** |
| DeepSeek-7B | 9.051 | 10.322 | 9.590 | **9.335** | **−0.255** |
| TinyLlama-1.1B | 10.480 | 14.692 | 12.917 | **10.931** | **−1.986** |

**7/7 wins over TurboQuant**, including the two heaviest-outlier models
(Llama-2-13B, DeepSeek-7B) that rotation-based quantization previously
owned. The result comes from **two orthogonal levers that are
super-additive on the hard-outlier tail**: neither the `combined_kvscore`
selector nor the exact K correction beats Turbo alone on DeepSeek /
Llama-2-13B, but stacked they clear it. This reverses the earlier
"TurboQuant Pareto-dominates CARE-KV" finding (which held for the 1st-order
K correction). Full ablation (current vs combined selector × linear vs
exact correction, NS=8/32/64) in `CLAUDE.md` §10.

> PPL / quality method — same-condition reimplementation of TurboQuant's
> QJL rotation, no CUDA kernels; prototype latency (see below). CARE-KV
> adds ~0.03× fp16 KV memory over base INT3 for the sparse residual.

## Repository layout

```
care_kv/                  # core package (imports as CARE_KV.care_kv)
  cache.py                # KV-head-indexed paged cache + KIVI side-buffer
  layer.py                # CARE-KV layer wrapper; base-quantizer dispatch
  attention.py            # CARE-KV attention; slot-based ΔO computation
  residual_router.py      # query-aware K/V scoring + top-k selection
  residual_store.py       # per-page candidate enumeration + storage
  quantizer.py            # per-group INT2/3/4 quant + dequant + packing
  kivi_helpers.py         # pure KIVI per-channel-K / per-token-V helpers
  llama_patch.py          # HF LlamaForCausalLM monkey-patch entrypoint
  baselines/              # same-condition adapters for SOTA comparison
    {fp16,basequant,carekv,kivi_style,...}_adapter.py
results/                  # paper-eval CSVs, summaries, figures, final_report
tools/                    # eval drivers + summarizer + figure scripts
scripts/                  # bash runners (paper matrix, per-section)
tests/                    # pytest + targeted invariant tests
```

## Setup

```bash
# Activate env (conda example)
conda activate vllm-carekv
export PYTHONPATH=$(realpath ..)   # parent of this repo
```

The package imports as `CARE_KV.care_kv` (not `care_kv`), so
`PYTHONPATH` must point at the **grandparent** directory of the source
tree — e.g. if this repo lives at `~/CARE_KV/care_kv/`, set
`PYTHONPATH=~`.

## Quick start

```bash
# 1. Verify the source compiles + tests pass (~70 s, no GPU needed)
python -m pytest -q tests/test_carekv_v2.py
python tests/test_kivi_dispatch.py

# 2. Run the paper-best CARE-KV on a synthetic 64-token prompt (~30 s on GPU)
python -c "
import torch
from transformers import AutoTokenizer
from CARE_KV.care_kv.baselines.carekv_adapter import CAREKVAdapter
from CARE_KV.care_kv.baselines.common import eval_ppl_synthetic
a = CAREKVAdapter(mode='fixed', bits=3)
m = a.setup_model('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
tok = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')
print('PPL:', eval_ppl_synthetic(m, tok, 64)[0])
"
```

## Reproduce the paper evaluation

Full matrix runner (one A6000, ~4 h):

```bash
PAPER_DIR=results/paper_eval_$(date +%Y%m%d_%H%M%S) \
  bash scripts/run_all_paper_eval.sh
```

Individual sections via env gates:

| flag | what runs |
|---|---|
| `RUN_PHASE_M=1` | routing-baseline ablation |
| `RUN_PHASE_N=1` | budget experiments (ratio vs absolute, sweeps, Pareto) |
| `RUN_PHASE_O=1` | adaptive read-budget mode |
| `RUN_PHASE_P=1` | same-condition SOTA direct comparison |
| `RUN_PHASE_Q=1` | CARE-KV on top of base quantizers (uniform + KIVI-style) |

Skip the heavy CARE-KV cells for a fast smoke (~15 min):

```bash
PAPER_DIR=results/paper_eval_20260529_015053 SKIP_HEAVY=1 \
  bash scripts/run_all_paper_eval.sh
```

After any run, regenerate the single-page report from the CSVs:

```bash
python tools/summarize_all_results.py results/paper_eval_20260529_015053
# → final_report.md, artifact_list.txt
```

## Paper-best configuration (locked)

Do not silently change these. Ablations should gate behind flags and
keep the defaults below.

```
BASE_BITS=3
CAREKV_PACKED_BASE=1
CAREKV_SCALE_QUANT=int8
CAREKV_PREFILL_MODE=carekv_stored
CAREKV_PREFILL_RESIDUAL_KIND=both
CAREKV_ROUTE_POLICY=joint
CAREKV_SCORE_NORMALIZE=1
CAREKV_CORRECTION_IMPL=vectorized   # combined selector is vectorized-only
CAREKV_K_CORRECTION_MODE=exact      # exact softmax renorm (not 1st-order)
CAREKV_KSCORE_LIVE=1                # combined_kvscore K+V selector
CAREKV_BUDGET_POLICY=uniform
STORE_ABS_K=2  STORE_ABS_V=4
READ_ABS_K=2   READ_ABS_V=2
```

The `exact` + `combined` config is the promoted paper-best (2026-07-15,
the headline table above). The prior `linear` / `cached` / current-selector
path stays reproducible via `CAREKV_K_CORRECTION_MODE=linear`,
`CAREKV_CORRECTION_IMPL=cached`, and unsetting `CAREKV_KSCORE_LIVE`. See
`CLAUDE.md` §2/§10 for the full cheat-sheet, the two-lever ablation, and the
rules for safe refactors.

## Status & limitations

- **Method-complete**: paper-best path validated end-to-end across 7
  LLaMA-arch models (1.1B–13B) at WT-2 NS=64, with the
  `READ=0 ≡ base_quant` invariant locked bit-exact in pytest (holds under
  both `exact` and `KSCORE_LIVE`).
- **Prototype latency**: prefill + correction are vectorized torch ops but
  still Python-driven per (layer, kv-head); no CUDA/Triton kernels. This is
  a **quality/memory method with no speed claim** — do not compare wall-clock
  to kernel-fused baselines.
- **Decode**: HF `use_cache=True` works (DynamicCache + open-page append)
  but with a fp16 dummy that inflates peak GPU memory ~2×.
- **Scope**: PPL (WikiText-2) at SL=512. TurboQuant is a same-condition QJL
  reimplementation. Downstream / long-context are separate (see `results/`).

## Key documents

- `final_report.md` — auto-generated single-page evaluation report.
- `summaries/` — hand-curated tables per experiment phase.
- `CLAUDE.md` — internal development guide (runtime knobs, pitfalls,
  paper-best lock).

## Citation

Paper in preparation. Use the GitHub URL for now:
`https://github.com/soeunida/care_kv`.
# CARE-KV-Sparse-Residual-Correction-for-Low-Bit-KV-Cache-Quantization
