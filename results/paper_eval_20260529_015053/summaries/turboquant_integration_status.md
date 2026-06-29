# TurboQuant integration status (base-quantizer-expansion)

> **Status as of 2026-06-04**: NOT integrated. Documented as
> `unsupported` in `ablations/base_quantizer_expansion_wt2_n4.csv`
> with the concrete blockers below. Per user spec, no simplified
> reimpl is implemented in this turn because the method's
> distinctive QJL (Quantized JL) residual stage is not unambiguously
> specified by the available sources.

## What the method is (from arXiv 2504.19874)

**TurboQuant**: "Online Vector Quantization with Near-optimal
Distortion Rate" — Google Research, ICLR 2026 submission. The core
algorithm has three stages:

1. **Random rotation** of K and V along head_dim. Data-oblivious
   (online), induces a concentrated Beta distribution across
   coordinates so per-coordinate scalar quant becomes near-optimal.
2. **Per-coordinate scalar quantization** — each channel of the rotated
   vector quantized independently with its optimal scalar quantizer.
3. **Two-stage inner-product correction**: MSE quantizer + a **1-bit
   QJL (Quantized JL) transform on the residual**, producing an
   *unbiased* inner-product estimator. This QJL stage is the
   distinctive contribution.

Headline numbers from the abstract: quality-neutral at **3.5 bits/channel**;
marginal degradation at **2.5 bits/channel**. ~2.7× from
information-theoretic optimum.

## Why we are NOT integrating this turn

### Blocker 1: no official Google code release
The paper is on arXiv but Google's official implementation has not
been released as of 2026-06-04. Per the standing rule, we do not
claim "official SOTA" without the official code being cloned, built,
imported, and run.

### Blocker 2: multiple community impls differ on the QJL specifics
Web search (2026-06-04) surfaced **four** community implementations:

| repo | claim |
|---|---|
| [0xSero/turboquant](https://github.com/0xSero/turboquant) | 3-bit keys, 2-bit values + Triton + vLLM integration |
| [back2matching/turboquant](https://github.com/back2matching/turboquant) | "First open-source" / pip install / HuggingFace drop-in |
| [vivekvar-dl/turboquant](https://github.com/vivekvar-dl/turboquant) | "First open-source" of arXiv:2504.19874 / 4-7× compression / pip install turbokv |
| [OnlyTerp/turboquant](https://github.com/OnlyTerp/turboquant) | "First open-source" of ICLR 2026 paper / 5× compression |

Three of these claim to be "first open-source". Their treatment of
the QJL residual stage (random seed handling, residual bit-width,
inner-product correction kernel) is not consistent across repos.
Picking one and labelling it "TurboQuant" would either silently bake
in that repo's choices, or commit us to a comparison that doesn't
faithfully represent the paper.

### Blocker 3: simplified reimpl would not be distinct from RotateKV
Without the QJL residual stage, "random rotation + per-channel
scalar quant" reduces to a randomized variant of our existing
`RotateKVStyleAdapter` (Walsh-Hadamard rotation + per-channel scalar
quant). Adding a 2nd rotation-based row to the table without the QJL
contribution would be redundant rather than informative.

## What a future integration would look like

When Google's official code is released (or one of the community
repos consolidates the QJL spec):

1. **Sidecar env if dep-heavy** — recent KV-cache quantization repos
   tend to pin specific torch/transformers/triton versions. Following
   the official KIVI integration pattern: clone under `external/`,
   pip install in env (or sidecar), gate behind `RUN_PHASE_R*=1` env
   flag.
2. **Adapter shape** — wrap upstream API in
   `baselines/official_turboquant.py:OfficialTurboQuantAdapter`,
   following the `OfficialKIVIAdapter` pattern.
3. **CARE-KV stacking** — TurboQuant's per-coordinate quant + QJL
   residual would need careful alignment with CARE-KV's residual
   slots. Likely a separate Phase Q-stacked-variant turn after the
   standalone TurboQuant lands.

## Notes

- This document is checked into the paper-eval dir alongside
  `official_kivi_cuda_integration_plan.md` so future turns can pick
  it up where we left off.
- The corresponding CSV row in
  `ablations/base_quantizer_expansion_wt2_n4.csv` carries
  `official_or_reimpl="unsupported"`, `ppl=0.0`, and a `notes` field
  pointing at this file.
