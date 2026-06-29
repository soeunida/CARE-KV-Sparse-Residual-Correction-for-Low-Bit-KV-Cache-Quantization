# Phase U — reconstruction-Pareto PPL validation

**Label:** `paper-ready candidate screen` (real `carekv_stored` path, WT-2 N=4, SL=128). Reconstruction columns are the idealized U-B basis; runtime store config + PPL are the deployed measurement. See header of the tool for the store-semantics caveat.

- model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`  BASE_BITS=3  max_pages=16
- fixed: PACKED_BASE=1, SCALE_QUANT=int8, PREFILL=carekv_stored, RESIDUAL_KIND=both, ROUTE=joint, SCORE_NORMALIZE=1, CORRECTION=cached, BUDGET_POLICY=uniform, READ_ABS=2/2 (except base_quant read=0/0).

| config | store policy | runtime K/V store spars | recon K | recon V | recon comb | resid mem KB | PPL | ΔPPL vs base | ΔPPL vs current | K_reads | V_reads | rt(s) | status |
|:-------|:-------------|:-----------------------:|------:|------:|------:|----------:|----:|-----:|-----:|------:|------:|----:|:--:|
| base_quant_INT3 | none (read=0) | 100%/100% | 0.0% | 0.0% | 0.0% | 0.0 | 16.1973 | 0.0 | 2.7355 | 0 | 0 | 19 | ok |
| per_page_current_SK2SV4 | per-page channel-group (paper-best) | 100%/100% | 16.8% | 33.4% | 25.1% | 72.0 | 13.4618 | -2.7355 | 0.0 | 641915 | 799877 | 1739 | ok |
| mag_SK1_SV4 | per-page, half K stored | 50%/100% | 8.7% | 33.4% | 21.0% | 60.0 | 13.5777 | -2.6196 | 0.1159 | 394276 | 1047516 | 3353 | ok |
| global_mag_kcg16_s12.5 | finer kcg16, low store (deployable approx) | 25%/50% | 21.5% | 18.7% | 20.1% | 48.0 | 14.2451 | -1.9522 | 0.7833 | 454636 | 987156 | 3311 | ok |
| global_mag_kcg16_s18.8 | finer kcg16, ~same-mem store (deployable approx) | 50%/50% | 30.7% | 26.7% | 28.7% | 72.0 | 14.6402 | -1.5571 | 1.1784 | 699976 | 741816 | 3780 | ok |
| high_recon_SK4_SV4 | per-page, store-saturated (SK4 caps at num_cg=2) | 100%/100% | 31.7% | 33.4% | 32.6% | 96.0 | 13.4618 | -2.7355 | 0.0 | 641915 | 799877 | 4473 | ok |

## Interpretation

**0. Sanity.** base_quant_INT3 PPL=16.1973. Current SK2SV4 PPL=13.4618 (ΔvsBase=-2.7355). Any CARE-KV row should sit at or below base_quant if the router fired (check K_reads/V_reads > 0).
**1. Does better reconstruction reduction also improve PPL? NO — reconstruction is not a reliable PPL proxy here.** The two highest-reconstruction candidates make this explicit: `high_recon_SK4_SV4` has the highest combined reconstruction (32.6%) but PPL **identical** to current (13.4618) because SK=4 caps to SK=2 at kcg=32 (store-saturated, unrealizable extra residual); and `global_mag_kcg16_s18.8` has the highest *distinct deployable* reconstruction (28.7%) yet the **worst** CARE-KV PPL (14.6402, +1.18 vs current). Among the deployable candidates, higher reconstruction reduction goes with *higher* (worse) PPL — the opposite of the hoped-for trend. (A naive linear ΔPPL-vs-reconstruction slope is ≈0 and misleading: it is dominated by the store-saturated SK4 duplicate, which equals current.)
**2. Does global_mag_kcg16_s18.8 beat current SK2SV4 at ~same memory?** PPL 14.6402 vs 13.4618 → NO — not lower PPL.
**3. Does global_mag_kcg16_s12.5 preserve PPL at lower memory?** PPL 14.2451 vs 13.4618 → NO — PPL degraded.
**4. Is magnitude-only selection good for PPL or only reconstruction?** mag_SK1_SV4 PPL=13.5777. U-B2 showed magnitude≈oracle for reconstruction; here we check whether that translates through the read-time Jacobian correction. Reducing stored K (SK1) raised PPL — store budget matters for what the router can read.
**5/7. Should the default change?** Best non-base PPL: per_page_current_SK2SV4 (13.4618). It does NOT beat current SK2SV4 (13.4618). **Keep SK2SV4 as paper-best; report the reconstruction Pareto as a diagnostic only (Q6).** Reconstruction-error reduction did not translate into a PPL win through the read-time correction at READ_ABS=2/2.

> Caveat / mechanism: the idealized U-B 'global magnitude' selection is not faithfully deployable in the current per-page channel-group store; candidates 4/5 are the closest deployable approximation (finer kcg=16). Decode-time sparsity is read-governed (READ_ABS=2/2 fixed here), which is largely orthogonal to the store-time reconstruction sparsity swept in U-B. The finer-kcg candidates lose on PPL because READ_ABS_K=2 reads 2 channel-group slots per query: at kcg=32 those cover all 64 head_dim channels, but at kcg=16 they cover only 32 of 64 — half the K residual is applied, so finer granularity *starves* the fixed read budget. The store-time reconstruction metric restores residuals at full precision and so never sees this read-budget coupling.

