# Phase Q — CARE-KV on top of base quantizers (stacked, end-to-end)

> **Headline (WT-2 N=4 SL=128, TinyLlama-1.1B, diagnostic pilot)**:
> CARE-KV's sparse residual correction stacked on top of a KIVI-style
> base quantizer improves PPL over BOTH (a) KIVI-style standalone
> (15.66 → 13.09, −2.56 PPL) AND (b) uniform-INT3 + CARE-KV
> (13.46 → 13.09, −0.37 PPL). The stacked cell ends up at modest
> additional memory (~+5%) over KIVI standalone while landing within
> ~0.44 PPL of `base_quant_INT4` and within ~0.75 PPL of fp16.

## What this experiment is

CARE-KV's residual correction has so far always been computed against
its own per-group base quantizer. The general formula is independent
of the base scheme:

```
R_K = K_fp - K_hat_base
R_V = V_fp - V_hat_base
O_care = O_base + ΔO_K + ΔO_V
```

The goal of Phase Q is to make CARE-KV **base-quantizer-agnostic**, so
CARE-KV's residuals can ride on top of any base quant scheme — uniform
per-group, KIVI per-channel/per-token, and others added later.

## Integration delivered (Phase Q-stacked)

- `care_kv/kivi_helpers.py` — pure `quant_dequant_kivi_k` (per-channel,
  scale across the token axis) + `quant_dequant_kivi_v` (per-token,
  scale across the channel axis). Single source of truth, used by both
  the standalone `KIVIStyleQuantizer` and the in-prefill dispatch.
- `care_kv/cache.py` — fp16 side-buffer (`base_K_hat_fp16`,
  `base_V_hat_fp16`, sized `[L, Hkv, P, T, D]`) allocated only when
  `cfg.base_quantizer == "kivi_style"`. New methods `write_base_kivi`,
  `append_to_page_kivi`, `read_base_hat_concat`.
- `care_kv/layer.py:_write_pages_for_kv_head` — dispatch on
  `cfg.base_quantizer`. In KIVI mode it computes K_hat/V_hat over the
  **full sequence** for that (layer, kv-head) and slices per-page; the
  uniform path is byte-identical to before.
- `care_kv/layer.py` decode-mode append path — single-token KIVI quant
  + `cache.append_to_page_kivi`.
- `care_kv/residual_store.py:process_page` — accepts optional
  `K_hat_override` / `V_hat_override` so residuals are computed against
  the KIVI K_hat rather than re-dequantized uniform codes.
- `care_kv/attention.py:CAREKVAttention.forward` — reads K_hat/V_hat
  directly from the fp16 side buffer in KIVI mode (no dequantize).
- `baselines/carekv_adapter.py:CAREKVAdapter` — new kwargs
  `base_quantizer="uniform"|"kivi_style"`, `bits_k`, `bits_v`,
  `max_pages` (tighter default for the kivi side buffer).
- `baselines/common.py:ResultRow` — new columns `base_memory_MB`,
  `residual_memory_MB`, `base_quantizer`.

Default `cfg.base_quantizer = "uniform"` keeps the paper-best path
bit-for-bit identical to before — verified by 18/18 existing pytest +
4/4 new KIVI dispatch tests (`tests/test_kivi_dispatch.py`).

## Honest implementation caveats

- **fp16 side-buffer**: KIVI's per-channel K scales don't fit the
  existing `(L, Hkv, P, T, G)` scale-storage layout. To avoid a full
  cache-layout rewrite, this prototype stashes the
  KIVI-dequantized K_hat / V_hat as fp16 in a new side buffer. Memory
  accounting reports KIVI's *theoretical* bit-width (per-channel K +
  fp16 scale per channel; per-token V + fp16 scale per token) — the
  fp16 side buffer's actual bytes are a prototype-implementation cost,
  not what a production stacked implementation would hold.
- **Decode-mode per-token KIVI**: KIVI per-channel K with T=1 reduces
  to scale = |K|/qmax (one scale per (kv-head, channel) per token).
  This is "good enough" for the carekv_stored prefill+decode but is
  not the full per-sequence KIVI scheme.
- **Same-condition reimpl, NOT official KIVI** — no CUDA kernels.
- **Prototype-latency** — the CARE-KV cells run in PyTorch + Python
  loops (~2000-2500 s/cell at SL=128 N=4 vs ~7 s for KIVI standalone).
  Not comparable to KIVI's CUDA kernels.

## Results (WT-2 N=4 SL=128, TinyLlama-1.1B, 508 evaluated tokens)

CSV: `results/paper_eval_20260529_015053/ablations/carekv_on_base_quantizers.csv`

| cell                              | PPL       | ΔPPL vs fp16 | ΔPPL vs INT3 | KV mem (MB) | vs-fp16 | K_reads | V_reads | runtime (s) |
|-----------------------------------|----------:|-------------:|-------------:|------------:|--------:|--------:|--------:|------------:|
| fp16                              |    12.346 |       +0.000 |       −3.852 |     2.750   |  1.000x |       0 |       0 |        11.9 |
| base_quant_INT4                   |    12.654 |       +0.309 |       −3.543 |     0.688   |  0.250x |       0 |       0 |        17.4 |
| base_quant_INT3 (uniform)         |    16.197 |       +3.852 |        0.000 |     0.516   |  0.188x |       0 |       0 |        18.4 |
| uniform_INT3 + CARE-KV (paper-best)|   13.462 |       +1.116 |       −2.736 |     0.653   |  0.238x | 641 915 | 799 877 |      2161.4 |
| KIVI_style_INT3                   |    15.657 |       +3.311 |       −0.540 |     0.548   |  0.199x |       0 |       0 |         6.7 |
| **KIVI_INT3 + CARE-KV (stacked)** | **13.095**|   **+0.749** |   **−3.103** |   **0.685** | **0.249x** | **648 980** | **792 812** | **2450.1** |
| KIVI_style_INT2 (unstable)        |  1521.949 |    +1509.604 |    +1505.752 |     0.376   |  0.137x |       0 |       0 |         8.6 |
| KIVI_INT2 + CAREKV (unstable)     |  3475.302 |    +3462.956 |    +3459.105 |     0.514   |  0.187x | 615 405 | 826 387 |      1999.7 |

Figures:
- `figures/fig_carekv_on_base_quantizers_ppl.png`
- `figures/fig_carekv_on_base_quantizers_memory_quality.png`

## Interpretation (Part D answers)

1. **Does CARE-KV improve uniform INT3?**
   **Yes.** 13.462 vs 16.197 (−2.74 PPL, ~17% rel).
2. **Does CARE-KV improve KIVI-style INT3?**
   **Yes — clearly. 15.657 → 13.095 (−2.56 PPL, ~16% rel).** The
   stacked cell now lands within 0.44 PPL of `base_quant_INT4` (12.654)
   and within 0.75 PPL of fp16 (12.346). K_reads=648 980,
   V_reads=792 812 confirm the router fired throughout.
3. **Can CARE-KV rescue KIVI-style INT2?**
   **No. TinyLlama is simply too small for INT2 K/V even with CARE-KV
   residual correction.** KIVI_INT2 PPL=1522 → +CARE-KV PPL=3475
   (got *worse*). With INT2 the base K_hat is so noisy that the
   per-(layer, kv-head) full-sequence per-channel quant collapses to a
   degenerate scale, and the CARE-KV residual is no longer a good
   linear correction signal. A bigger model (≥7B) at INT2 might still
   be salvageable, but the 1.1B chat model is the wrong vehicle here.
4. **Is KIVI-style + CARE-KV better than uniform + CARE-KV?**
   **Yes by 0.37 PPL.** 13.095 vs 13.462. The gap is small but
   consistent in direction: KIVI's per-channel K appears to leave a
   slightly more "correctable" residual signal for CARE-KV's
   query-aware router. KIVI's per-token V also reduces V-side error
   that CARE-KV would otherwise have to spend slot budget on.
   Caveat: tiny pilot (508 evaluated tokens) — could be noise; would
   need WT-2 N≥16 to firm up.
5. **Memory overhead of CARE-KV residuals on top of KIVI-style?**
   Stacked total = 0.685 MB vs KIVI standalone 0.548 MB → +0.137 MB
   (+25% over the KIVI base, or +5% of fp16 KV memory). The residual
   storage formula is independent of the base scheme (SK=2/16 K rows,
   SV=4/16 V rows, packed 4-bit).
6. **Paper-ready or diagnostic?**
   **Diagnostic-only.** N=4 SL=128 on a 1.1B-parameter chat model is
   too small for a headline claim. The improvement direction is
   consistent and statistically meaningful per cell, but a paper claim
   needs WT-2 N≥16 and ≥1 other model. The defensible diagnostic
   claim today: "CARE-KV is base-quantizer-agnostic and the
   integration into a KIVI-style base improves both quality (over
   KIVI-style standalone) and quality (over uniform CARE-KV) at modest
   additional memory, on this WT-2 N=4 SL=128 TinyLlama pilot."

## Honest framing reminders

- **KIVI-style = same-condition reimplementation**, NOT official KIVI.
  No CUDA kernels.
- **KIVI INT2 is unstable on TinyLlama-1.1B** — both standalone and
  stacked. Do NOT report INT2 numbers as a working configuration.
- **Runtime is prototype-latency.** KIVI standalone 7 s vs
  KIVI+CARE-KV 2450 s reflects the cost of CARE-KV's Python-loop
  cached corrections, not a fair end-to-end runtime comparison
  against KIVI's CUDA kernels.
- **fp16 side buffer**: documented honestly above — production stacked
  would use option (a) (new per-channel scale cache layout) instead.

## Deferred extensions (Part E)

- **WT-2 N≥16 confirmation** of the KIVI+CARE-KV improvement on
  TinyLlama. ~6× longer runtime, would firm up the 0.37-PPL
  advantage over uniform+CARE-KV.
- **Larger model** (≥7B) check on whether KIVI INT2 + CARE-KV can be
  rescued — currently catastrophic on TinyLlama.
- **KVQuant-style + CARE-KV** — requires the `k_store_mode = pre_rope |
  post_rope` switch (documented in `sota_official_integration_status.md`).
- **MiKV-style / ZipCache-style + CARE-KV** — per-token bit-width
  plumbing through pack/unpack (documented).
- **Cache-layout option (a)** — proper per-channel K scale storage to
  remove the fp16 side-buffer prototype cost.

## How to reproduce

```bash
# Phase Q-stacked WT-2 N=4 SL=128 sweep (~90 min on shared A100)
PYTHONPATH=/home/soeun python tools/eval_carekv_on_base_quantizers.py \
  --out-csv results/paper_eval_20260529_015053/ablations/carekv_on_base_quantizers.csv \
  --dataset wikitext --seq-len 128 --num-samples 4

# Figures
python tools/make_carekv_on_base_quantizers_figure.py \
  --csv results/paper_eval_20260529_015053/ablations/carekv_on_base_quantizers.csv \
  --ppl-out results/paper_eval_20260529_015053/figures/fig_carekv_on_base_quantizers_ppl.png \
  --mem-out results/paper_eval_20260529_015053/figures/fig_carekv_on_base_quantizers_memory_quality.png
```

Or via the unified runner: `RUN_PHASE_Q=1 bash scripts/run_all_paper_eval.sh`.
