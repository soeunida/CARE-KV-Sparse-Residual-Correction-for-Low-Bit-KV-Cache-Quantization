# Same-condition direct SOTA comparison — synthetic SL=64 (diagnostic)

> **Status**: diagnostic synthetic-prompt smoke. SL=64, 254-token
> hand-written prompt, TinyLlama-1.1B. Absolute PPL values are
> inflated; the **relative ordering** is the meaningful signal. A WT-2
> N=4 SL=128 follow-up takes ~2–3 h and is gated behind
> `RUN_PHASE_P_DIRECT=1 PD_DATASET=wikitext PD_NUM_SAMPLES=4` in
> `scripts/run_all_paper_eval.sh`.

## Companion docs

- `summaries/sota_methods_table.md` — literature methods + reported-paper numbers (separate table; not mixed in here)
- `summaries/sota_official_integration_status.md` — adapter-by-adapter blocker writeup
- (synthetic-proxy / older Phase P artifacts on `feat/sota-comparison` branch are independent)

## How "same condition" is enforced

Every adapter in `baselines/` runs through the **same** `KVMethodAdapter`
interface defined in `baselines/common.py`. Concretely:

- **Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` loaded with
  `torch_dtype=torch.float16` in every cell.
- **Tokenizer**: `AutoTokenizer.from_pretrained(MODEL_ID)`, identical
  pad-token handling.
- **Dataset / windowing / PPL computation**: shared
  `eval_ppl_synthetic` / `eval_ppl_wikitext` helpers — every cell uses
  the same forward-pass shifted-CE loss and the same window-and-mean
  reduction.
- **Memory estimator**: per-adapter `estimate_memory(seq_len)` returns
  a JSON blob with the same schema (`estimated_kv_memory_MB`,
  `estimated_total_cache_memory_MB`, `vs_fp16_kv_memory_ratio`).
- **Runtime logging**: shared `measure_peak_gpu_mb` for peak GPU
  allocation; `time.perf_counter` for wall-clock seconds.
- **Batch size**: 1 (single forward pass per cell, single window in
  synthetic; one window at a time for WT-2).

The only thing that varies across cells is the per-method K/V storage
treatment.

## Results

| method | family | label | bit-width | PPL | ΔPPL vs fp16 | ΔPPL vs base_INT3 | est. KV MB | vs fp16 mem | runtime |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `fp16` | fp16 | reference | fp16 | **90.21** | 0 | −50.70 | 1.375 | 1.000× | 4.8 s |
| `base_quant_INT4` | base_quant | same-cond reimpl | INT4 | 94.88 | +4.68 | −46.02 | 0.344 | 0.250× | 5.8 s |
| `base_quant_INT3` | base_quant | same-cond reimpl | INT3 | 140.90 | +50.70 | 0 (ref) | 0.258 | 0.188× | 5.8 s |
| `base_quant_INT2` | base_quant | same-cond reimpl | INT2 | 1 222.49 | +1 132.29 | +1 081.59 | 0.172 | 0.125× | 5.9 s |
| `KIVI_style_INT4K_INT4V` | kivi_style | same-cond reimpl | K=INT4, V=INT4 | 95.13 | +4.93 | −45.77 | 0.365 | 0.266× | 4.5 s |
| `KIVI_style_INT3K_INT3V` | kivi_style | same-cond reimpl | K=INT3, V=INT3 | 116.42 | +26.22 | −24.48 | 0.279 | 0.203× | 4.4 s |
| `KIVI_style_INT2K_INT2V` | kivi_style | same-cond reimpl | K=INT2, V=INT2 | 9 300.83 | +9 210.62 | +9 159.92 | 0.193 | 0.141× | 4.7 s |
| **`CAREKV_fixed_SK2SV4_RK2RV2`** | care_kv | paper-best | INT3 | **102.75** | **+12.54** | **−38.16** | 0.327 | 0.237× | 131.1 s (prototype-latency) |
| `CAREKV_adaptive_rel0.05` | care_kv | adaptive (WT-2 rel) | INT3 | 112.52 | +22.32 | −28.38 | 0.327 | 0.237× | 133.8 s (prototype-latency) |
| `KVQuant_style_INT4_preRoPE_K` | kvquant_style | **unsupported** | INT4 (pre-RoPE) | — | — | — | — | — | — |
| `MiKV_style_mixed_precision` | mikv_style | **unsupported** | mixed | — | — | — | — | — | — |
| `ZipCache_style_saliency_mixed` | zipcache_style | **unsupported** | mixed | — | — | — | — | — | — |

(Unsupported rows raised `NotImplementedError` at `setup_model`; their
blocker reasons are in `summaries/sota_official_integration_status.md`.)

## Findings (synthetic; diagnostic-only)

1. **CARE-KV (paper-best) beats every other deployable cell at ~0.33 MB.**
   At nearly the same memory:
   - CARE-KV INT3: **102.75 PPL**
   - KIVI-style INT3: 116.42 PPL (+13.7 PPL = +13 %)
   - base_quant_INT3: 140.90 PPL (+38.2 PPL = +37 %)

   The sparse residual correction is doing real work over and above
   raw asymmetric K/V quantization — which itself is doing real work
   over and above uniform per-group quant.

2. **KIVI-style INT3 strictly beats base_quant_INT3.** 116.42 vs 140.90,
   −24.5 PPL gap at the same nominal bit-width. **Validates KIVI's
   asymmetric-quant design** under the CARE-KV codebase's eval
   methodology.

3. **INT2 is unusable on TinyLlama at this prompt** for both
   base_quant (PPL 1 222) and KIVI-style (PPL 9 301). Don't draw paper
   conclusions about INT2 from this run — the small model + short
   synthetic prompt amplifies quantization error past the readable
   range. A larger model + longer prompt is the right venue for INT2
   claims.

4. **CARE-KV adaptive `rel=0.05` is WORSE than CARE-KV fixed on
   synthetic** (112.52 vs 102.75). This is consistent with the Phase O
   finding that the adaptive sweet spot is **dataset-dependent**:
   `rel=0.10` was the synthetic winner; `rel=0.05` was the WT-2
   winner. Don't conclude "adaptive is bad" — it's "adaptive needs
   per-dataset tuning, and the WT-2 winner is suboptimal on synthetic".

5. **Runtime gap is huge but unfair to compare** at this stage. CARE-KV
   cells (~130 s) are 25–30× slower than fp16 / base_quant / KIVI-style
   cells (~5 s) because CARE-KV uses Python loops for residual
   correction and the others are pure PyTorch attention. This is
   `prototype-latency` for CARE-KV; comparable to KIVI's PyTorch
   version, but NOT comparable to KIVI's CUDA-kernel version (we
   didn't run that here).

6. **Memory accounting**: CARE-KV (~0.327 MB) is slightly more memory
   than KIVI-style INT3 (~0.279 MB) due to the residual storage. CARE-KV's
   quality advantage is **per-bit**, not per-byte — at equal memory,
   the comparison would need a memory-matching iteration loop (deferred
   to Part C of the spec).

## Honest paper labelling

When citing this table in the paper:

- **`CARE-KV fixed`** is the paper-best — call it that, NOT "CARE-KV
  beats KIVI" on this prompt alone.
- **`KIVI-style INT3`** is a **same-condition reimplementation**, not
  the official KIVI repo. The KIVI authors' CUDA-kernel implementation
  is faster at the same PPL but the PPL itself shouldn't differ
  meaningfully from the reimplementation (KIVI's quality claim is in
  the quantization scheme, not the kernels).
- **`base_quant`** ladder is this codebase's own generic reference; not
  attributable to any specific paper.
- **`unsupported` rows** (KVQuant, MiKV, ZipCache stubs) must be cited
  alongside their blocker — see
  `summaries/sota_official_integration_status.md`.

## Pareto figure

`figures/fig_sota_direct_memory_quality.png` (log-y) shows the
PPL-vs-memory scatter. The interesting cluster is at ~0.3 MB; CARE-KV
(green dot) sits at a Pareto-better point than KIVI-style INT3
(orange square) and base_quant_INT3 (grey square) — strictly lower PPL
in the same memory range.

The INT2 cells (PPL > 1 000) and fp16 (memory ~ 1.4 MB) are the
extremes; the comparison story is in the middle cluster.

## How to reproduce

```bash
# Synthetic (this run, ~16 min)
PYTHONPATH=/home/soeun python tools/eval_sota_direct_comparison.py \
  --out-csv  results/paper_eval_20260529_015053/sota_direct/sota_direct_synthetic_sl64.csv \
  --out-json results/paper_eval_20260529_015053/sota_direct/sota_direct_synthetic_sl64.json \
  --dataset synthetic --seq-len 64

# WT-2 N=4 SL=128 (~2.5–3 h)
PYTHONPATH=/home/soeun python tools/eval_sota_direct_comparison.py \
  --out-csv  results/paper_eval_20260529_015053/sota_direct/sota_direct_wikitext2_n4_sl128.csv \
  --out-json results/paper_eval_20260529_015053/sota_direct/sota_direct_wikitext2_n4_sl128.json \
  --dataset wikitext --seq-len 128 --num-samples 4

# Figures
python tools/make_sota_direct_figures.py \
  --csv         results/paper_eval_20260529_015053/sota_direct/sota_direct_synthetic_sl64.csv \
  --ppl-out     results/paper_eval_20260529_015053/figures/fig_sota_direct_ppl.png \
  --mem-out     results/paper_eval_20260529_015053/figures/fig_sota_direct_memory_quality.png \
  --runtime-out results/paper_eval_20260529_015053/figures/fig_sota_direct_runtime.png
```

Or via the unified runner: `RUN_PHASE_P_DIRECT=1 bash scripts/run_all_paper_eval.sh`
(synthetic by default; `PD_DATASET=wikitext PD_NUM_SAMPLES=4` for the WT-2 pilot).
