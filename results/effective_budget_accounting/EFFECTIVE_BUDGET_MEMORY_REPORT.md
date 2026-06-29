# CARE-KV Effective-Budget Memory Accounting & Selector Report

SK/SV is **not** a universally fixed deployment budget. This report distinguishes three analyses:

1. **Fixed SK2SV4 controlled analysis** — for explaining the head_dim overhead trend (every cell costed at the same SK2SV4).
2. **Effective deployment budget** — what the adaptive selector actually requires per cell.
3. **PPL–memory budget-quality Pareto** — accuracy vs KV bytes across budgets, using only real measured PPLs.

## Audit: are K residual buffers allocated/used under Vdom?

Code audit of the V-only (Vdom) path:
- **Store** (`residual_store.py`): `kind="v"` → `use_k=False` → `k_budget=0` → `alloc_k_slot()` is never called → **zero K residual slots written**.
- **Read** (`residual_router.py`): `kind="v"` → `_resolve_read_budgets` forces `bk=0`, and the K-candidate loop is guarded by `kind in {"k","both"}` → **zero K residual slots read**.
- **But** (`cache.py`): the K residual arena (`k_residual_buf`/`k_residual_scale`) was **pre-allocated** at construction sized by `(store_abs_k+store_abs_v)`, so a V-only deployment still reserved a full K arena = dead memory. The as-run experiments stored `kind=both` (SK2SV4) and routed Vdom at eval, so K residuals were physically present but unused for Vdom cells.

**Finding: K residual buffers are allocated but unused in Vdom** → the effective Vdom budget is **SK0SV4**.

**Optimization added** (flag-gated, default off → paper-best preserved): `CAREKV_VDOM_OPTIMIZED=1` drops the K arena to a 1-slot stub (~99.7% K-arena reduction in a unit check) and makes `alloc_k_slot()` raise as an audit guard. Verified lossless by `tests/test_carekv_v2.py::test_vdom_optimized_kstore_audit` (V-only route byte-identical with vs without the K arena); full suite (20 tests) green.

## 1. Fixed SK2SV4 controlled analysis (head_dim trend)

Every cell costed at SK2SV4. Residual overhead ratio = `(SK+SV)·1.13/(2·head_dim·bits/8)` is arch-only, so it falls with head_dim — this is the controlled explanation table (kept intact).

| head_dim | head_dim_group | n_cells | fixed_residual_overhead_vs_base_pct | fixed_residual_frac_of_total_pct | fixed_saving_vs_fp16_pct |
| --- | --- | --- | --- | --- | --- |
| 128 | d97_160 | 27 | 7.0625 | 5.7495 | 76.9680 |

## 2. Effective deployment budget after the adaptive selector

Mapping: `BaseQuant_INT3`/`skip_near_lossless`→**SK0SV0**, `Vdom`→**SK0SV4**, `KV`→**SK2SV4**.

Distribution:

| selected_correction | effective_store_budget | n_cells |
| --- | --- | --- |
| BaseQuant_INT3 | SK0SV0 | 16 |
| KV | SK2SV4 | 6 |
| Vdom | SK0SV4 | 5 |

Per-cell effective budget (real-arch cells):

| model_id | seq_len | selected_correction | fixed_total_kv_MB | effective_store_budget | effective_total_kv_MB | effective_residual_frac_of_total_pct | effective_total_kv_MB_reduction_vs_fixed_pct | gate_a_pass | gate_b_pass | carekv_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mistralai/Mistral-7B-v0.3 | 512 | KV | 14.7405 | SK2SV4 | 14.7405 | 5.7495 | 0.0000 | True | True | real |
| mistralai/Mistral-7B-v0.3 | 256 | Vdom | 7.3703 | SK0SV4 | 7.1635 | 3.9436 | 2.8052 | True | True | real |
| 01-ai/Yi-6B | 128 | KV | 1.8426 | SK2SV4 | 1.8426 | 5.7495 | 0.0000 | True | True | real |
| 01-ai/Yi-6B | 256 | BaseQuant_INT3 | 3.6851 | SK0SV0 | 3.2500 | 0.0000 | 11.8076 | True | True | real |
| deepseek-ai/deepseek-llm-7b-base | 512 | Vdom | 55.2769 | SK0SV4 | 53.7263 | 3.9436 | 2.8052 | True | True | real |
| openlm-research/open_llama_7b_v2 | 512 | Vdom | 58.9620 | SK0SV4 | 57.3080 | 3.9436 | 2.8052 | True | True | real |
| deepseek-ai/deepseek-llm-7b-base | 128 | KV | 13.8192 | SK2SV4 | 13.8192 | 5.7495 | 0.0000 | True | True | real |
| deepseek-ai/deepseek-llm-7b-base | 256 | BaseQuant_INT3 | 27.6384 | SK0SV0 | 24.3750 | 0.0000 | 11.8076 | True | False | blocked |
| 01-ai/Yi-6B | 512 | KV | 7.3703 | SK2SV4 | 7.3703 | 5.7495 | 0.0000 | True | True | real |
| Qwen/Qwen2-7B | 128 | BaseQuant_INT3 | 1.6122 | SK0SV0 | 1.4219 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| Qwen/Qwen2.5-7B | 128 | BaseQuant_INT3 | 1.6122 | SK0SV0 | 1.4219 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| Qwen/Qwen2.5-7B | 256 | BaseQuant_INT3 | 3.2245 | SK0SV0 | 2.8438 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| Qwen/Qwen2.5-7B | 512 | BaseQuant_INT3 | 6.4490 | SK0SV0 | 5.6875 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| Qwen/Qwen2-7B | 256 | BaseQuant_INT3 | 3.2245 | SK0SV0 | 2.8438 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| Qwen/Qwen2-7B | 512 | BaseQuant_INT3 | 6.4490 | SK0SV0 | 5.6875 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| facebook/opt-6.7b | 256 | KV | 29.4810 | SK2SV4 | 29.4810 | 5.7495 | 0.0000 | True | True | real |
| EleutherAI/pythia-6.9b | 128 | BaseQuant_INT3 | 14.7405 | SK0SV0 | 13.0000 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| facebook/opt-6.7b | 128 | BaseQuant_INT3 | 14.7405 | SK0SV0 | 13.0000 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| mistralai/Mistral-7B-v0.3 | 1024 | BaseQuant_INT3 | 29.4810 | SK0SV0 | 26.0000 | 0.0000 | 11.8076 | True | True | real |
| mistralai/Mistral-7B-v0.3 | 128 | Vdom | 3.6851 | SK0SV4 | 3.5817 | 3.9436 | 2.8052 | True | True | real |
| EleutherAI/pythia-6.9b | 256 | BaseQuant_INT3 | 29.4810 | SK0SV0 | 26.0000 | 0.0000 | 11.8076 | False | True | blocked_architecture_port |
| openlm-research/open_llama_7b_v2 | 256 | KV | 29.4810 | SK2SV4 | 29.4810 | 5.7495 | 0.0000 | True | True | real |
| openlm-research/open_llama_7b_v2 | 128 | BaseQuant_INT3 | 14.7405 | SK0SV0 | 13.0000 | 0.0000 | 11.8076 | True | True | real |
| 01-ai/Yi-6B | 1024 | Vdom | 14.7405 | SK0SV4 | 14.3270 | 3.9436 | 2.8052 | True | True | real |
| openlm-research/open_llama_7b_v2 | 1024 | BaseQuant_INT3 | 117.9240 | SK0SV0 | 104.0000 | 0.0000 | 11.8076 | True | False | blocked |
| deepseek-ai/deepseek-llm-7b-base | 1024 | BaseQuant_INT3 | 110.5537 | SK0SV0 | 97.5000 | 0.0000 | 11.8076 | True | False | blocked |
| mistralai/Mistral-7B-v0.3 | 1024 | BaseQuant_INT3 | 29.4810 | SK0SV0 | 26.0000 | 0.0000 | 11.8076 | True | True | real |

_Note: `fixed_*` columns (controlled SK2SV4) are preserved alongside `effective_*` columns in `effective_budget_all_rows.csv`. Cells where the selector chose Vdom or BaseQuant deploy with strictly less KV memory than the fixed SK2SV4 accounting implies._

## 3. PPL–memory budget-quality Pareto

Memory is analytical (batch=1, packed int8, INT3). PPL is **real measured** only: SK0SV0 = BaseQuant INT3, SK0SV4 = `calibration_ppl_vdom`, SK2SV4 = `calibration_ppl_kv`. **SK0SV2 / SK4SV4 have no measured PPL rows** and appear memory-only (`ppl_source=no_ppl_row`) — never faked.

Example — `01-ai/Yi-6B` SL128 N4 (all three measured PPLs):

| budget | total_kv_MB | saving_vs_fp16_pct | ppl | ppl_source | is_effective_choice | pareto_status |
| --- | --- | --- | --- | --- | --- | --- |
| SK0SV0 | 1.6250 | 79.6875 | 9.2370 | measured_basequant_int3 | False | frontier |
| SK0SV4 | 1.7909 | 77.6141 | 9.2703 | measured_calibration_vdom | False | dominated |
| SK2SV4 | 1.8426 | 76.9680 | 9.1891 | measured_calibration_kv | True | frontier |

Full ladder in `budget_quality_pareto.csv` (49 measured-PPL points, 54 memory-only points).

## Honesty / preservation

- Analytical estimator memory ≠ measured GPU peak; clean-allocation validation lives in the residual-overhead report.
- Fixed SK2SV4 columns are preserved for the controlled head_dim analysis; effective columns are added, not substituted.
- Failed/OOM/blocked/collapsed CARE-KV cells are preserved in `effective_budget_all_rows.csv` with their `carekv_status` and selector decision (e.g. Qwen `no_valid_carekv_candidate` / `blocked_architecture_port`). They map to SK0SV0 effective but are flagged, not silently counted as wins.
- The Vdom→SK0SV4 claim is backed by the code audit + the lossless optimized-path test, not assumed.
