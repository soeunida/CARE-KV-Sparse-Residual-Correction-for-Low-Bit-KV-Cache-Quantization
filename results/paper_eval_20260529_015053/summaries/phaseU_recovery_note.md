# Phase U — recovery note (shared-checkout contamination incident)

**Date of recovery:** 2026-06-05
**Recovery worktree:** `~/CARE_KV/care_kv_phaseU_recover`
**Recovery branch:** `feat/residual-reconstruction-pareto-recover` (based on local
`master` = 71772aa, the fast-forwarded `origin/master`).

## What happened

Phase U residual-reconstruction work was done on branch
`feat/residual-reconstruction-pareto` and kept **uncommitted (untracked)** per the
instruction not to commit until confirmed. While the Phase U PPL-validation run was
executing in the background (~4.6 h), another terminal operated on the **same shared
checkout**: it switched branches (`feat/residual-reconstruction-pareto` → `master` →
`feat/base-quantizer-expansion`) and ran a `git clean` / `git stash -u`, which deleted
untracked files mid-run. Because nothing was committed, almost all Phase U work was
untracked and was removed.

## What was lost (not recoverable from disk; source/tables held only in chat context)

- Phase U tool scripts:
  - `tools/u_residual_recon_engine.py`
  - `tools/eval_residual_reconstruction_pareto.py`        (Part A)
  - `tools/eval_residual_reconstruction_store_sweep.py`   (U-B1)
  - `tools/eval_residual_selection_policy_sweep.py`       (U-B2)
  - `tools/eval_residual_reconstruction_granularity_sweep.py` (U-B3)
  - `tools/build_residual_reconstruction_pareto.py`       (U-B4)
  - `tools/eval_reconstruction_pareto_ppl_validation.py`  (PPL — restored, see below)
- Phase U artifacts (CSVs / MDs / figures) for Part A and U-B1–U-B4:
  - `ablations/residual_reconstruction_pareto_layer_summary.csv`, `…store_sweep.csv`,
    `residual_selection_policy_sweep.csv`, `…granularity_sweep.csv`,
    `residual_reconstruction_pareto.csv`
  - matching `summaries/*.md` and `figures/fig_residual_*.png` /
    `fig_*_error_reduction_per_kb.png`
  - intermediate capture cache `_u_capture.pt` (already deleted before incident)

## What survived (written after the clean, then backed up)

The Phase U **PPL-validation** outputs survived because they were written at the very
end of the run, after the clean. They were backed up to:

**Backup path:** `/tmp/care_kv_phaseU_backup_1780621486/`
- `reconstruction_pareto_ppl_validation.csv`
- `reconstruction_pareto_ppl_validation.md`  (corrected interpretation)
- `fig_reconstruction_pareto_ppl_validation.png`
- `u_ppl_validation.log`

A full dirty-tree snapshot (all untracked files at incident time, including the other
terminal's work) was also taken at `/tmp/care_kv_current_dirty_snapshot_1780621772/`.

## Recovered files (restored into this worktree)

- `ablations/reconstruction_pareto_ppl_validation.csv`            (from backup)
- `summaries/reconstruction_pareto_ppl_validation.md`            (from backup, corrected)
- `figures/fig_reconstruction_pareto_ppl_validation.png`         (from backup)
- `logs/u_ppl_validation.log`                                    (from backup)
- `tools/eval_reconstruction_pareto_ppl_validation.py`           (exact original source
  restored from chat context + a recovery banner; re-running needs the lost U-B CSVs for
  the reconstruction columns — see banner)

## Phase U PPL validation table (real `carekv_stored` path; TinyLlama, WT-2, SL=128, N=4)

| config | recon comb | resid mem KB | PPL | ΔvsBase | ΔvsCurrent | K_reads | V_reads |
|:-------|------:|----:|----:|----:|----:|------:|------:|
| base_quant_INT3 | 0.0% | 0.0 | 16.1973 | 0.0000 | +2.7355 | 0 | 0 |
| **per_page_current_SK2SV4** | 25.1% | 72.0 | **13.4618** | **−2.7355** | 0.0000 | 641915 | 799877 |
| mag_SK1_SV4 | 21.0% | 60.0 | 13.5777 | −2.6196 | +0.1159 | 394276 | 1047516 |
| global_mag_kcg16_s12.5 | 20.1% | 48.0 | 14.2451 | −1.9522 | +0.7833 | 454636 | 987156 |
| global_mag_kcg16_s18.8 | 28.7% | 72.0 | 14.6402 | −1.5571 | +1.1784 | 699976 | 741816 |
| high_recon_SK4_SV4 | 32.6% | 96.0 | 13.4618 | −2.7355 | 0.0000 | 641915 | 799877 |

## Conclusion

**Reconstruction-Pareto candidates did NOT improve PPL.** No candidate beats the current
configuration. The two highest-reconstruction candidates make the decoupling explicit:
`high_recon_SK4_SV4` (32.6% combined reconstruction, the highest) is PPL-identical to
current because SK=4 is capped to SK=2 at kcg=32 (store-saturated, unrealizable extra
residual), and `global_mag_kcg16_s18.8` (28.7%, the highest *distinct deployable*
reconstruction) has the **worst** CARE-KV PPL (14.6402). Finer kcg=16 starves the fixed
READ budget (READ_ABS_K=2 slots cover all 64 head-dim channels at kcg=32 but only 32/64
at kcg=16), so the store-time reconstruction gain does not survive the read-time
correction. Reconstruction-error reduction is a useful **diagnostic** for where residual
energy lives, but is **not** a reliable PPL proxy at READ_ABS=2/2.

## Decision

**Paper-best remains unchanged: SK=2, SV=4, RK=2, RV=2** (`per_page_current`,
PPL 13.4618). The Phase U-B reconstruction Pareto is recorded as a diagnostic only; no
candidate is promoted, and no WT-2 N=16 confirmation is warranted.
