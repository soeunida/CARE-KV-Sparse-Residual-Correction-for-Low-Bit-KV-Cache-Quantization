# Same-condition direct comparison — WikiText-2 N=4 SL=128 (6-cell)

## Headline

**Under the same TinyLlama/WikiText-2 pilot setting, CARE-KV INT3 improves
PPL over base_quant_INT3 and a KIVI-style INT3 reimplementation while
using modest additional cache memory. CARE-KV narrows the quality gap
between INT3 and INT4, reaching within 0.81 PPL of base_quant_INT4 at
slightly lower estimated KV memory.**

### Important caveats (read before citing)

- This is a **same-condition direct comparison**, NOT an official SOTA
  runtime comparison.
- The KIVI-style rows are a **same-condition reimplementation of KIVI's
  K/V quantization scheme**, NOT the official KIVI repo. Official KIVI
  ships custom CUDA kernels that are NOT included here.
- KVQuant, MiKV, and ZipCache are listed as **unsupported in this turn**
  with concrete blockers — see
  `summaries/sota_official_integration_status.md`.
- **Do NOT compare CARE-KV prototype runtime against official
  CUDA-kernel methods.** CARE-KV runs at prototype-latency (Python-loop
  residual correction); KIVI-style and base_quant cells here also run
  under PyTorch only, so the quality axis is fair but the throughput
  axis is not informative against kernel-accelerated SOTA.

> **Sourcing note**: the underlying numbers were produced by the prior
> 12-cell direct-comparison run
> (`sota_direct_wikitext2_n4_sl128.csv`) and were *filtered* to the
> 6-cell subset specified in this report. **No re-run was performed** —
> every cell uses the exact same model load, tokenizer, dataset
> windowing, PPL code, and memory estimator as the source run, so
> filtering is equivalent to re-running this subset. Total
> source-run wall-clock was 3 167 s ≈ 53 min.

> **Sample size**: N=4 windows × SL=128 = 508 evaluated tokens. Treat
> PPL gaps under ~0.2 as within noise.

## Setup (identical for every cell)

- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (fp16)
- Dataset: WikiText-2 `wikitext-2-raw-v1` test, N=4 non-overlapping SL=128 windows = 508 tokens
- Tokenizer, dataset loader, windowing, shifted-CE PPL, peak GPU memory: shared `baselines/common.py` helpers
- Cache memory estimator: per-adapter, same schema (`estimated_kv_memory_MB`, `vs_fp16_kv_memory_ratio`)
- Adapter framework: `baselines/{fp16,basequant,kivi_style,carekv}_adapter.py`

## Results

| method | paper label | bit-width | PPL | ΔPPL vs fp16 | ΔPPL vs INT3 base | est. KV MB | vs fp16 mem | runtime |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `fp16`                          | **reference**                    | fp16          | **12.346** | 0       | −3.852 | 2.750 | 1.000× | 12.4 s |
| `base_quant_INT4`                | baseline (CARE-KV base_quant)    | INT4          | 12.654 | +0.309  | −3.543 | 0.688 | 0.250× | 16.8 s |
| `base_quant_INT3`                | baseline (CARE-KV base_quant)    | INT3          | 16.197 | +3.852  |  0 (ref) | 0.516 | 0.188× | 16.7 s |
| `KIVI_style_INT3K_INT3V`         | same-condition reimplementation  | K=INT3, V=INT3 | 15.657 | +3.311  | −0.540 | 0.548 | 0.199× | 6.0 s |
| `KIVI_style_INT2K_INT2V`         | same-condition reimpl (**INSTABLE at INT2**) | K=INT2, V=INT2 | 1 521.95 | +1 509.6 | +1 505.8 | 0.376 | 0.137× | 7.0 s |
| **`CAREKV_fixed_SK2SV4_RK2RV2`** | **CARE-KV (paper-best)**         | INT3          | **13.462** | **+1.116** | **−2.736** | 0.653 | 0.237× | 1 555.7 s (prototype-latency) |

## Findings (honest trade-off framing)

1. **CARE-KV INT3 has the lowest PPL among the INT3-budget cells**, at
   modestly higher memory than the other INT3 cells:
   - CARE-KV INT3:       13.46 PPL @ 0.65 MB  (24 % of fp16)
   - KIVI-style INT3:    15.66 PPL @ 0.55 MB  (20 % of fp16) — **+2.20 PPL, −18 % memory** vs CARE-KV
   - base_quant_INT3:    16.20 PPL @ 0.52 MB  (19 % of fp16) — **+2.74 PPL, −21 % memory** vs CARE-KV

   CARE-KV is **not Pareto-dominant** here — it trades ~18–21 % more
   memory (and dramatically more prototype-latency runtime) for a 2.2–2.7
   PPL quality improvement. Whether that trade is "worth it" is a
   user/deployment decision; this table just reports both axes
   honestly.

2. **CARE-KV INT3 sits within 0.81 PPL of base_quant_INT4** (13.46 vs
   12.65) at **slightly less estimated KV memory** (0.65 vs 0.69 MB).
   This is the strongest defensible framing: CARE-KV at INT3 narrows
   the INT3→INT4 quality gap to <1 PPL on this pilot.

3. **KIVI-style INT3 beats base_quant_INT3 by 0.54 PPL** on WT-2 — small
   but consistent win for asymmetric K/V quantization (per-channel K +
   per-token V). Confirms KIVI's design helps on real text under
   same-condition eval too, just at a smaller margin than the synthetic
   prompt suggested.

4. **INT2 is unusable on TinyLlama at this sequence length.** KIVI-style
   INT2 hits PPL 1 521.95 (vs INT3's 15.66) — a >100× degradation.
   INT2 results need a larger model + longer sequence to be meaningful;
   we report the data point honestly and mark it `INSTABLE`.

5. **Runtime asymmetry: prototype-latency, NOT a head-to-head against
   official SOTA.** CARE-KV is ~1 556 s vs ~6–17 s for the others.
   This is a Python-loop residual correction; both CARE-KV and the
   KIVI-style cells run under PyTorch only in this harness (no custom
   kernels for either). Comparing wall-clock against KIVI's official
   CUDA implementation would be unfair both ways — quality numbers
   from a PyTorch reimpl should be comparable, but throughput numbers
   should not.

## Fairness disclosures

- **`KIVI_style_INT3K_INT3V`** and **`KIVI_style_INT2K_INT2V`** are
  **same-condition reimplementations** of KIVI's per-channel-K +
  per-token-V quantization scheme, implemented inside this codebase
  via monkey-patching `apply_rotary_pos_emb` and `v_proj`. They do
  **not** use the official KIVI CUDA kernels — quality is faithful
  to the published quant scheme, runtime is not. Do NOT cite as
  "official KIVI."
- **`base_quant_INT3`** and **`base_quant_INT4`** are CARE-KV's
  generic base-quantization path. They're labelled "baseline" — not
  attributable to any specific paper.
- **`CAREKV_fixed_SK2SV4_RK2RV2`** is this codebase's paper-best
  fixed-budget config (no adaptive read budget). The WT-2-tuned
  adaptive variant (`rel=0.05`, PPL 12.93) is excluded from this
  6-cell table on purpose; see
  `sota_direct_wikitext2_n4_sl128.md` for the full 12-cell view.

## Excluded from this comparison

- **KVQuant** (pre-RoPE K storage required) → `unsupported`
- **MiKV** (per-token bit-width plumbing required) → `unsupported`
- **ZipCache** (per-token bit-width plumbing required) → `unsupported`
- See `summaries/sota_official_integration_status.md` for the
  blocker-by-blocker writeup.

## Memory–quality scatter (NOT a Pareto-dominance claim)

`figures/fig_sota_direct_memory_quality.png` (log-y) shows the
PPL-vs-memory scatter. With only 6 cells the picture is clean:

- Far right (high memory): **fp16** at 2.75 MB, PPL 12.35.
- Top-left (unusable): **KIVI-style INT2** at 0.38 MB, PPL > 1 500.
- Bottom cluster (deployable): each cell is a different point on a
  **memory–quality trade-off curve**, not a strict dominance:
  - `base_quant_INT3`   — lowest memory (0.52 MB), worst PPL among deployable INT3 cells (16.20)
  - `KIVI-style INT3`   — slightly more memory (0.55 MB), modestly better PPL (15.66)
  - `CAREKV INT3`       — most memory in the INT3 band (0.65 MB), best PPL in the INT3 band (13.46)
  - `base_quant_INT4`   — highest memory in the cluster (0.69 MB), best PPL in the cluster (12.65)

  Going `base_quant_INT3 → KIVI-style INT3 → CARE-KV INT3 → base_quant_INT4`
  trades memory for quality monotonically. CARE-KV is the **best point
  in the INT3 memory band** at the cost of being the most
  memory-expensive INT3 cell.

## Defensible paper claim (use this wording verbatim)

> **Under the same TinyLlama / WikiText-2 pilot setting, CARE-KV INT3
> improves PPL over base_quant_INT3 and a KIVI-style INT3
> reimplementation while using modest additional cache memory. CARE-KV
> narrows the quality gap between INT3 and INT4, reaching within 0.81
> PPL of base_quant_INT4 at slightly lower estimated KV memory.**

Do NOT frame as "CARE-KV Pareto-dominates KIVI" (it doesn't — it uses
more memory). Do NOT frame as "CARE-KV beats official KIVI" without
the "same-condition reimplementation" qualifier (we didn't run
official KIVI). Do NOT frame as a runtime win (this is
prototype-latency vs PyTorch-only baselines, not vs CUDA kernels).

## How to reproduce

```bash
# Source: 12-cell direct comparison (53 min wall-clock)
PYTHONPATH=/home/soeun python tools/eval_sota_direct_comparison.py \
  --out-csv  results/paper_eval_20260529_015053/sota_direct/sota_direct_wikitext2_n4_sl128.csv \
  --out-json results/paper_eval_20260529_015053/sota_direct/sota_direct_wikitext2_n4_sl128.json \
  --dataset wikitext --seq-len 128 --num-samples 4

# Filter to this 6-cell subset (no re-run needed if the source CSV exists)
python3 -c "
import csv
KEEP = ['fp16','base_quant_INT4','base_quant_INT3',
        'KIVI_style_INT3K_INT3V','KIVI_style_INT2K_INT2V',
        'CAREKV_fixed_SK2SV4_RK2RV2']
src = 'results/paper_eval_20260529_015053/sota_direct/sota_direct_wikitext2_n4_sl128.csv'
dst = 'results/paper_eval_20260529_015053/sota_direct/sota_direct_wikitext2_n4.csv'
rows = {r['method_name']: r for r in csv.DictReader(open(src))}
filtered = [rows[n] for n in KEEP if n in rows]
import csv
w = csv.DictWriter(open(dst, 'w'), fieldnames=list(filtered[0]))
w.writeheader(); [w.writerow(r) for r in filtered]
"

# Figures
python tools/make_sota_direct_figures.py \
  --csv         results/paper_eval_20260529_015053/sota_direct/sota_direct_wikitext2_n4.csv \
  --ppl-out     results/paper_eval_20260529_015053/figures/fig_sota_direct_ppl.png \
  --mem-out     results/paper_eval_20260529_015053/figures/fig_sota_direct_memory_quality.png \
  --runtime-out results/paper_eval_20260529_015053/figures/fig_sota_direct_runtime.png
```
