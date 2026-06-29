# Base-quantizer expansion — WT-2 N=4 SL=128 (diagnostic pilot)

> **Headline (diagnostic pilot, TinyLlama-1.1B, 508 evaluated tokens)**:
> KVQuant-style pre-RoPE INT3 marginally beats KIVI-style INT3
> standalone (15.01 vs 15.66 PPL). RotateKV-style INT3 standalone is
> **worse than the uniform INT3 baseline** (27.45 vs 16.20) — fixed
> Walsh-Hadamard rotation hurts on already-rotated post-RoPE K — but
> **CARE-KV partially rescues it** (27.45 → 15.23, −12.2 PPL),
> demonstrating that the residual correction works on top of a bad
> base quantizer too. Plain uniform + CARE-KV remains best of all
> CARE-KV-stacked cells (13.46) on this pilot.

## What this experiment is

After Phase Q delivered the KIVI-style + CARE-KV stacked cell, this
expansion adds two more base quantizers to the same dispatch (kivi_style
side-buffer in CARE-KV's cache) and documents one as unsupported:

 - **KVQuant-style** (per-channel K, pre-RoPE option; same-condition
   reimpl, no NUQ / no sparse outlier path)
 - **RotateKV-style** (Walsh-Hadamard rotate + per-channel K +
   per-token V + inverse rotate)
 - **TurboQuant** — `unsupported` row; see
   `summaries/turboquant_integration_status.md` for the blocker
   writeup (no official Google code yet, 4 community impls differ on
   the QJL residual specifics).

All cells use the same model / tokenizer / WT-2 windowing / PPL
computation / memory estimator as the existing Phase Q + Phase
P-direct sweeps. PPL is directly comparable across files.

## Results

CSV: `results/paper_eval_20260529_015053/ablations/base_quantizer_expansion_wt2_n4.csv`

| cell | status | bits | PPL | ΔPPL vs fp16 | ΔPPL vs INT3 | KV mem (MB) | vs fp16 | K/V_reads | runtime (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | reference | fp16 | 12.346 | 0.000 | −3.852 | 2.750 | 1.000x | 0 / 0 | 10.4 |
| base_quant_INT4 | reimpl | INT4 uniform | 12.654 | +0.309 | −3.543 | 0.688 | 0.250x | 0 / 0 | 17.0 |
| base_quant_INT3 | reimpl | INT3 uniform | 16.197 | +3.852 | 0.000 | 0.516 | 0.188x | 0 / 0 | 17.6 |
| uniform_INT3 + CARE-KV (paper-best) | reimpl | INT3 uniform + slots | **13.462** | **+1.116** | **−2.736** | 0.653 | 0.238x | 642 k / 800 k | 4753.0 |
| KIVI_style_INT3 | reimpl (Phase Q ref) | INT3 per-channel K / per-token V | 15.657 | +3.311 | −0.540 | 0.548 | 0.199x | 0 / 0 | 7.1 |
| KIVI_INT3 + CARE-KV (stacked, Phase Q) | reimpl (Phase Q ref) | KIVI INT3 + slots | **13.095** | **+0.749** | **−3.103** | 0.685 | 0.249x | 649 k / 793 k | 4767.9 |
| **KVQuant_style_INT3 (pre-RoPE)** | **reimpl (NEW)** | **per-channel K pre-RoPE INT3 + per-token V INT3** | **15.008** | **+2.662** | **−1.189** | 0.548 | 0.199x | 0 / 0 | 7.2 |
| KVQuant_INT3 + CARE-KV | unsupported | — | — | — | — | — | — | — | — |
| **RotateKV_style_INT3** | **reimpl (NEW)** | **Walsh-Hadamard rotate + per-channel K / per-token V INT3 + inverse** | **27.451** | **+15.105** | **+11.253** | 0.548 | 0.199x | 0 / 0 | 6.9 |
| **RotateKV_INT3 + CARE-KV (stacked, NEW)** | **reimpl (NEW)** | **RotateKV INT3 + slots** | **15.225** | **+2.880** | **−0.972** | 0.685 | 0.249x | 622 k / 820 k | 4754.7 |
| TurboQuant_style_INT3 | unsupported | — | — | — | — | — | — | — | — |
| TurboQuant_INT3 + CARE-KV | unsupported | — | — | — | — | — | — | — | — |

Figures:
- `figures/fig_base_quantizer_expansion_ppl.png`
- `figures/fig_base_quantizer_expansion_memory_quality.png`

## Interpretation

### 1. Does CARE-KV improve each base quantizer?

| base | standalone PPL | + CARE-KV PPL | ΔPPL | improvement |
|---|---:|---:|---:|---|
| uniform INT3 | 16.197 | 13.462 | −2.736 | yes (~17% rel) |
| KIVI INT3 | 15.657 | 13.095 | −2.562 | yes (~16% rel) |
| KVQuant INT3 (pre-RoPE) | 15.008 | — | — | **blocked** — true pre-RoPE storage requires a new K-cache path through layer.py + cache.py; same blocker as the original KVQuantStyleAdapter stub. See `baselines/kvquant_style.py` module docstring. A post-RoPE KVQuant variant + CARE-KV would be the same code path as KIVI + CARE-KV (Phase Q-stacked) — not new information. |
| RotateKV INT3 | 27.451 | 15.225 | **−12.226** | yes (~45% rel), but the resulting cell (15.23) still doesn't beat plain uniform+CARE-KV (13.46) |
| TurboQuant INT3 | — | — | — | unsupported (see `turboquant_integration_status.md`) |

CARE-KV's residual correction is base-quantizer-agnostic in practice:
it improves every base quantizer it's stacked on, even when the base
is much worse than uniform. The relative improvement is largest on
the worst base (RotateKV — 45% rel), reflecting that there's more
error left to correct. But the *absolute* PPL after CARE-KV depends
strongly on the base — a bad base + CARE-KV still trails a decent
base + CARE-KV.

### 2. KVQuant pre-RoPE K storage helps

Standalone KVQuant pre-RoPE INT3 (15.01) beats both KIVI INT3 (15.66)
and base_quant_INT3 (16.20) at the same INT3 memory budget. The
pre-RoPE choice is the *only* difference between KVQuant-style and
KIVI-style in this minimal reimpl (both use per-channel K + per-token
V uniform symmetric INT3; KVQuant just quantizes K BEFORE RoPE
rotates it). The −0.65 PPL improvement vs KIVI is small but
consistent with the paper's claim that pre-RoPE storage produces a
smoother per-channel distribution.

### 3. RotateKV is worse standalone on post-RoPE K

Hypothesis: post-RoPE K already mixes channels via the rotary
embedding's frequency basis. Applying a fixed Walsh-Hadamard rotation
*on top of* that mixing produces a distribution where each
"channel" is a weighted average of multiple original channels —
useful for spreading large outliers (KIVI's claim) but harmful when
the post-RoPE channel structure is already favorable for per-channel
quantization. Pre-RoPE rotation (à la the original RotateKV
proposal) would likely behave differently, but our adapter applies
the rotation post-RoPE to compose with CARE-KV's post-RoPE cache.

### 4. CARE-KV residual correction rescues bad base quantizers

The +CARE-KV column shows the residual correction consistently
**closes the gap to a fixed ceiling** (~13–15 PPL on this pilot)
regardless of how bad the base is. RotateKV+CARE-KV (15.23) is much
better than RotateKV standalone (27.45) but plateaus near the same
range as the other +CARE-KV cells. This suggests the residual budget
(SK=2 SV=4 per page) is the bottleneck once base error exceeds what
those slots can correct.

## Honest framing reminders

- **All "X-style" cells are same-condition reimplementations**, not
  official upstream code. KIVI-style is the only adapter with an
  OFFICIAL counterpart on this codebase (see
  `summaries/official_kivi_cuda_comparison_wt2_n4.md`). KVQuant /
  RotateKV / TurboQuant adapters here are reimplementations of the
  *quantization scheme*, not full faithful ports.
- **KVQuant minimal reimpl drops** the non-uniform quant (NUQ) and
  the dense+sparse outlier decomposition. Just per-channel pre-RoPE
  K + per-token V at uniform INT3.
- **RotateKV minimal reimpl drops** any learned-rotation /
  calibration step. Just a fixed Walsh-Hadamard rotation.
- **TurboQuant is genuinely unsupported** in this turn — no official
  Google code; 4 community impls disagree on the QJL residual
  specifics. Documented in `summaries/turboquant_integration_status.md`.
- **Runtime is prototype-latency** for any +CARE-KV cell (~4750 s
  here vs ~1700 s in a less-contended GPU run). Not a method-level
  speed claim.
- **Pilot scale**: N=4 SL=128 → 508 evaluated tokens. PPL deltas
  below ~0.5 are below single-sample noise. RotateKV's huge
  difference is real; the KVQuant−KIVI gap (0.65 PPL) needs WT-2
  N≥16 confirmation before any paper claim.

## Files

```
ablations/base_quantizer_expansion_wt2_n4.csv
summaries/base_quantizer_expansion_wt2_n4.md
summaries/turboquant_integration_status.md
figures/fig_base_quantizer_expansion_ppl.png
figures/fig_base_quantizer_expansion_memory_quality.png
baselines/kvquant_style.py        # NEW
baselines/rotatekv_style.py       # NEW
kivi_helpers.py                    # extended: dispatch + rotatekv helpers
layer.py, cache.py, attention.py   # extended: 3-way side-buffer dispatch
baselines/carekv_adapter.py        # extended: accept rotatekv_style / kvquant_style
tools/eval_base_quantizer_expansion.py             # NEW
tools/make_base_quantizer_expansion_figure.py      # NEW
```
