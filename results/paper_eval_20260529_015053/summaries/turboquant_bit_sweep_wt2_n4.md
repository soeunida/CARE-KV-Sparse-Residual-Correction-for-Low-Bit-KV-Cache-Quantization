# Part A — TurboQuant-style INT4/INT3/INT2 bit sweep

> **Headline (WT-2 N=4 SL=128, TinyLlama, diagnostic pilot)**: A faithful
> TurboQuant-style reimplementation (random orthonormal rotation +
> per-coordinate scalar quant + **QJL 1-bit residual inner-product
> correction**) shows the QJL stage is decisive at **INT3** — it cuts PPL
> from **24.04 (rotation+quant only) → 14.25 (+QJL)**, beating BaseQuant
> INT3 (16.20). At INT4 the base is already near-lossless so there is little
> room (12.65 → 12.68). At INT2 everything collapses. **NOT official
> TurboQuant** — no official code exists; this is a same-condition reimpl.

## Cells (16 rows)

| method | PPL | ΔvsFP16 | Δvs-same-bit-Base | est KV MB | residual MB | status |
|---|---|---|---|---|---|---|
| fp16 | 12.3457 | 0 | — | 2.75 | 0 | reference |
| BaseQuant INT4 | 12.6544 | +0.31 | 0 | 0.688 | 0 | reimpl |
| BaseQuant INT3 | 16.1973 | +3.85 | 0 | 0.516 | 0 | reimpl |
| BaseQuant INT2 | 334.65 | +322 | 0 | 0.344 | 0 | reimpl |
| **TurboQuant INT4 +QJL** | 12.6791 | +0.33 | +0.02 | 0.913 | 0.193 | reimpl |
| TurboQuant INT4 noQJL | 12.9484 | +0.60 | +0.29 | 0.720 | 0 | reimpl |
| **TurboQuant INT3 +QJL** | **14.2509** | +1.91 | **−1.95** | 0.741 | 0.193 | reimpl |
| TurboQuant INT3 noQJL | 24.0397 | +11.69 | +7.84 | 0.548 | 0 | reimpl |
| TurboQuant INT2 +QJL | 1239.85 | — | +905 | 0.569 | 0.193 | reimpl |
| TurboQuant INT2 noQJL | 5033.18 | — | +4698 | 0.376 | 0 | reimpl |
| TurboQuant INT4/3/2 + CARE-KV | — | — | — | — | — | **unsupported** |
| uniform INT3 + CARE-KV | 13.4618 | +1.12 | −2.74 | 0.653 | 0.138 | reimpl |
| KIVI INT3 + CARE-KV | 13.0948 | +0.75 | −3.10 | 0.685 | 0.138 | reimpl |
| KVQuant-preRoPE INT3 + CARE-KV | 13.1004 | +0.75 | −3.10 | 0.685 | 0.138 | reimpl |

Figures: `fig_turboquant_bit_sweep_ppl.png`, `fig_turboquant_bit_sweep_memory_quality.png`.
CSV: `ablations/turboquant_bit_sweep_wt2_n4.csv`.

## Implementation & honesty

- **Same-condition reimplementation, NOT official.** No official TurboQuant
  code exists (4 community repos disagree on QJL specifics — see
  `turboquant_integration_status.md`). All rows labelled accordingly.
- **Three faithful stages**: seeded random orthonormal rotation (QR of a
  Gaussian — the data-oblivious rotation TurboQuant specifies, distinct from
  RotateKV's fixed Hadamard) → per-coordinate scalar quant → **QJL**: store
  1-bit `sign(S·r)` + `‖r‖` per key, estimate `⟨q,r⟩ ≈ ‖r‖·√(π/2)·⟨sign(Sr),Sq⟩/m`.
- **QJL numerically validated** before the eval: the estimator is unbiased
  (bias≈0) and reduces inner-product MSE for `m ≥ 2·head_dim` (break-even at
  m≈128 for d=64; below that the 1-bit estimator's variance exceeds the
  residual). Eval uses **m = 2·head_dim = 128**.
- **TurboQuant + CARE-KV is `unsupported`**: QJL is a *score-level
  inner-product* correction, not a *reconstruction* base quantizer, so
  CARE-KV's reconstruction residual slots cannot be stacked on the QJL
  estimate without redefining both methods. (uniform/KIVI/KVQuant bases ARE
  reconstruction quantizers and stack fine — the 3 anchors.)

## The five questions

1. **Is BaseQuant INT4 already near-lossless?** **Yes** — 12.65 vs fp16 12.35
   (+0.31, +2.5%). On TinyLlama INT4 is essentially free.
2. **Does TurboQuant INT4 have room to improve?** **Little** — TurboQuant INT4
   +QJL (12.68) ≈ BaseQuant INT4 (12.65). QJL still helps the *rotation*
   baseline (12.95 → 12.68) but the headroom over plain INT4 is ~0.
3. **Does TurboQuant help more at INT3 or INT2?** **INT3, decisively.** QJL
   cuts INT3 from 24.04 → 14.25 (−9.8), and TurboQuant INT3 (14.25) **beats**
   BaseQuant INT3 (16.20) by 1.95. At INT2 QJL helps (5033 → 1240) but the
   cell is unusable either way.
4. **Does CARE-KV help TurboQuant?** Not evaluable as a stack (unsupported,
   above). The comparable *reconstruction*-base CARE-KV stacks (uniform/KIVI/
   KVQuant INT3 + CARE-KV ≈ 13.1–13.5) are slightly better than TurboQuant
   INT3 +QJL (14.25) at similar memory.
5. **Is INT4 too easy under SL=128 N=4?** **Yes** — BaseQuant INT4 is
   near-lossless, so INT4 does not discriminate methods at this scale. (Part
   B shows INT4 is *not* near-lossless on a real 7B: +1.52 PPL — the regime
   where low-bit quant actually matters.)

## Memory note

QJL costs a 1-bit-per-projection sketch + a per-key norm: at m=128 that is
~0.19 MB (≈+35% over the INT3 rotation base 0.55 MB). So the INT3 +QJL win
(−1.95 PPL vs BaseQuant INT3) comes at a real memory cost — landing near the
CARE-KV stacks' memory but with slightly worse PPL.

*Diagnostic pilot (N=4, SL=128). Needs N≥16 confirmation.*
