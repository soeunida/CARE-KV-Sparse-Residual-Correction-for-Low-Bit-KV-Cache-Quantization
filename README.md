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
to compute ΔO_K, ΔO_V. The base quantizer is pluggable — CARE-KV runs
on top of its own per-group quant **or** a KIVI-style (per-channel K,
per-token V) reimplementation.

## Headline result (diagnostic pilot)

WikiText-2 N=4, SL=128, TinyLlama-1.1B-Chat-v1.0, 508 evaluated tokens:

| cell | PPL | KV memory | vs INT3 |
|---|---:|---:|---:|
| fp16 | 12.346 | 2.75 MB | −3.85 |
| base_quant_INT4 | 12.654 | 0.69 MB | −3.54 |
| base_quant_INT3 (uniform) | 16.197 | 0.52 MB | 0.00 |
| uniform_INT3 + CARE-KV | 13.462 | 0.65 MB | −2.74 |
| KIVI_style_INT3 | 15.657 | 0.55 MB | −0.54 |
| **KIVI_INT3 + CARE-KV (stacked)** | **13.095** | **0.69 MB** | **−3.10** |

KIVI_INT3 + CARE-KV stacked **beats both** KIVI_INT3 standalone
(−2.56 PPL) and uniform+CARE-KV (−0.37 PPL) at modest additional
memory.

> Diagnostic-only — pilot scale. Needs WT-2 N≥16 and ≥1 other model
> before a paper claim. Same-condition reimplementation of KIVI's
> K/V quantization scheme (no CUDA kernels).

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
CAREKV_CORRECTION_IMPL=cached
CAREKV_BUDGET_POLICY=uniform
STORE_ABS_K=2  STORE_ABS_V=4
READ_ABS_K=2   READ_ABS_V=2
```

See `CLAUDE.md` for the full runtime-knobs cheat-sheet, the
historical pitfalls list, and the rules for safe refactors.

## Status & limitations

- **Method-complete**: paper-best path is validated end-to-end with the
  `READ=0 ≡ base_quant` invariant locked in pytest.
- **Prototype latency**: prefill + correction are Python loops over
  (layer, kv-head, token). Wall-clock is 1500–2500 s per CARE-KV cell
  at TinyLlama SL=128 N=4. Not comparable to CUDA-kernel methods.
- **Decode**: HF `use_cache=True` works (DynamicCache + open-page append)
  but with a fp16 dummy that inflates peak GPU memory ~2×.
- **Models**: TinyLlama-1.1B-Chat-v1.0 + JackFram/llama-160m verified.
  Larger models need HF auth or a larger local cache.
- **KIVI-style integration** uses an fp16 K_hat/V_hat side-buffer in
  the cache (memory accounting reports KIVI's theoretical bits). A
  production stacked implementation would add a per-channel scale
  storage path instead.

## Key documents

- `final_report.md` — auto-generated single-page evaluation report.
- `summaries/` — hand-curated tables per experiment phase.
- `CLAUDE.md` — internal development guide (runtime knobs, pitfalls,
  paper-best lock).

## Citation

Paper in preparation. Use the GitHub URL for now:
`https://github.com/soeunida/care_kv`.
# CARE-KV-Sparse-Residual-Correction-for-Low-Bit-KV-Cache-Quantization
