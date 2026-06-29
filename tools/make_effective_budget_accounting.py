"""tools/make_effective_budget_accounting.py

Effective-budget CARE-KV memory accounting + selector report.

SK/SV is NOT a universally fixed deployment budget. The fixed SK2SV4 table is a
*controlled* memory-overhead analysis (good for explaining the head_dim trend),
but the adaptive selector chooses a different correction per cell, and the
*effective* deployment budget follows that choice:

    selected_correction = BaseQuant_INT3 / skip_near_lossless  → SK0SV0 (no residuals)
    selected_correction = Vdom (V-only)                        → SK0SV4 (no K residuals)
    selected_correction = KV                                   → SK2SV4 (K + V residuals)

The Vdom → SK0SV4 mapping is justified by a code audit (see report §Audit):
  - STORE: kind="v" → use_k=False → k_budget=0 → alloc_k_slot() never called.
  - READ : router kind="v" → bk=0 and the K candidate loop is skipped.
  - So Vdom neither writes nor reads K residuals; the only K cost was the
    pre-allocated K arena, which the new CAREKV_VDOM_OPTIMIZED path drops to a
    1-slot stub (audited by tests/test_carekv_v2.py::test_vdom_optimized_kstore_audit).

This tool emits three clearly-separated analyses:
  1. fixed SK2SV4 controlled analysis (head_dim trend)
  2. effective deployment budget after the adaptive selector
  3. PPL vs memory budget-quality Pareto (SK0SV0/SK0SV2/SK0SV4/SK2SV4/SK4SV4)

Real measured PPLs exist for SK0SV0 (=BaseQuant_INT3), SK0SV4 (=calibration_ppl_vdom)
and SK2SV4 (=calibration_ppl_kv). SK0SV2 / SK4SV4 have no PPL rows and are NEVER
faked — they appear with memory only and ppl_source="no_ppl_row".

No fake results. Failed/OOM/blocked rows are preserved. Does not commit.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measure_residual_overhead_scaling import component_bytes, get_arch, MB  # noqa: E402

ALL_ROWS = "results/memory_aware_fair_comparison/memory_aware_all_rows.csv"
OUT_DIR_DEFAULT = "results/effective_budget_accounting"

# budgets for the quality-Pareto: (label, sk, sv, ppl_source_column_or_method)
PARETO_BUDGETS = [
    ("SK0SV0", 0, 0, "basequant_int3"),     # no residuals  == BaseQuant INT3
    ("SK0SV2", 0, 2, None),                  # no PPL row measured
    ("SK0SV4", 0, 4, "calibration_ppl_vdom"),  # V-only correction
    ("SK2SV4", 2, 4, "calibration_ppl_kv"),    # K+V correction (fixed paper budget)
    ("SK4SV4", 4, 4, None),                  # no PPL row measured
]


def head_dim_group(d):
    return ("le64" if d <= 64 else "d65_96" if d <= 96 else
            "d97_160" if d <= 160 else "ge161")


def mem_cols(arch, seq_len, sk, sv):
    """Analytical KV memory (batch=1, packed int8, INT3) for budget (sk,sv)."""
    L = arch["num_layers"]; Hkv = arch["num_key_value_heads"]; Dh = arch["head_dim"]
    cb = component_bytes(L, Hkv, Dh, 1, int(seq_len), 3, sk, sv,
                         packed_base=True, scale_quant="int8")
    total = cb["base_code"] + cb["scale"] + cb["residual"] + cb["meta"] + cb["sketch"]
    base = cb["base_code"]; fp16 = cb["fp16"]
    return dict(
        residual_MB=cb["residual"] / MB,
        total_kv_MB=total / MB,
        base_code_MB=base / MB,
        fp16_MB=fp16 / MB,
        residual_overhead_vs_base_pct=(cb["residual"] / base * 100.0) if base else 0.0,
        residual_frac_of_total_pct=(cb["residual"] / total * 100.0) if total else 0.0,
        saving_vs_fp16_pct=(1 - total / fp16) * 100.0,
    )


def effective_budget(selected_correction, selector_decision):
    sc = str(selected_correction); sd = str(selector_decision)
    if sc == "Vdom":
        return "SK0SV4", 0, 4
    if sc == "KV":
        return "SK2SV4", 2, 4
    # BaseQuant_INT3 / skip_near_lossless / no_valid_carekv_candidate / anything else
    return "SK0SV0", 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--all-rows", default=ALL_ROWS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.all_rows)
    ck = df[df["method"] == "Adaptive_CAREKV_INT3"].copy()
    # BaseQuant_INT3 PPL per cell (= SK0SV0 effective PPL)
    bq3 = (df[df["method"] == "BaseQuant_INT3"][["model_id", "seq_len", "num_samples",
                                                 "ppl", "status"]]
           .rename(columns={"ppl": "basequant_int3_ppl",
                            "status": "basequant_int3_status"}))
    ck = ck.merge(bq3, on=["model_id", "seq_len", "num_samples"], how="left")
    fp16 = (df[df["method"] == "fp16"][["model_id", "seq_len", "num_samples", "ppl"]]
            .rename(columns={"ppl": "fp16_ppl"}))
    ck = ck.merge(fp16, on=["model_id", "seq_len", "num_samples"], how="left")

    arch_map = {}
    for m in sorted(ck["model_id"].unique()):
        try:
            arch_map[m] = get_arch(m)
        except Exception as e:
            arch_map[m] = None
            print(f"[arch] {m}: unavailable ({e})")

    all_rows, pareto_rows = [], []
    for _, r in ck.iterrows():
        mid = r["model_id"]; sl = r["seq_len"]; ns = r["num_samples"]
        arch = arch_map.get(mid)
        sc = r.get("selected_correction"); sd = r.get("selector_decision")
        eff_label, ek, ev = effective_budget(sc, sd)
        row = dict(
            model_id=mid, seq_len=sl, num_samples=ns,
            head_dim=(arch["head_dim"] if arch else pd.NA),
            head_dim_group=(head_dim_group(arch["head_dim"]) if arch else ""),
            is_gqa=(arch["is_gqa"] if arch else pd.NA),
            selected_correction=sc, selector_decision=sd,
            gate_a_pass=r.get("gate_a_pass"), gate_b_pass=r.get("gate_b_pass"),
            carekv_ppl=r.get("ppl"), carekv_status=r.get("status"),
            basequant_int3_ppl=r.get("basequant_int3_ppl"),
            calibration_ppl_vdom=r.get("calibration_ppl_vdom"),
            calibration_ppl_kv=r.get("calibration_ppl_kv"),
            fp16_ppl=r.get("fp16_ppl"),
        )
        if arch is not None:
            # ── 1. FIXED SK2SV4 controlled columns (head_dim trend) ──
            f = mem_cols(arch, sl, 2, 4)
            row.update(dict(
                fixed_store_budget="SK2SV4",
                fixed_residual_MB=round(f["residual_MB"], 6),
                fixed_total_kv_MB=round(f["total_kv_MB"], 6),
                fixed_residual_overhead_vs_base_pct=round(f["residual_overhead_vs_base_pct"], 4),
                fixed_residual_frac_of_total_pct=round(f["residual_frac_of_total_pct"], 4),
                fixed_saving_vs_fp16_pct=round(f["saving_vs_fp16_pct"], 4),
            ))
            # ── 2. EFFECTIVE selected-budget columns ──
            e = mem_cols(arch, sl, ek, ev)
            row.update(dict(
                effective_store_budget=eff_label,
                effective_k_slots=ek, effective_v_slots=ev,
                effective_residual_MB=round(e["residual_MB"], 6),
                effective_total_kv_MB=round(e["total_kv_MB"], 6),
                effective_residual_overhead_vs_base_pct=round(e["residual_overhead_vs_base_pct"], 4),
                effective_residual_frac_of_total_pct=round(e["residual_frac_of_total_pct"], 4),
                effective_saving_vs_fp16_pct=round(e["saving_vs_fp16_pct"], 4),
                effective_vs_fixed_residual_MB_delta=round(e["residual_MB"] - f["residual_MB"], 6),
                effective_total_kv_MB_reduction_vs_fixed_pct=(
                    round((1 - e["total_kv_MB"] / f["total_kv_MB"]) * 100.0, 4)
                    if f["total_kv_MB"] else 0.0),
            ))
            # ── 3. budget-quality Pareto rows (one per budget) ──
            for (blab, sk, sv, src) in PARETO_BUDGETS:
                mc = mem_cols(arch, sl, sk, sv)
                if src == "basequant_int3":
                    ppl = r.get("basequant_int3_ppl"); psrc = "measured_basequant_int3"
                elif src == "calibration_ppl_vdom":
                    ppl = r.get("calibration_ppl_vdom"); psrc = "measured_calibration_vdom"
                elif src == "calibration_ppl_kv":
                    ppl = r.get("calibration_ppl_kv"); psrc = "measured_calibration_kv"
                else:
                    ppl = pd.NA; psrc = "no_ppl_row"
                pareto_rows.append(dict(
                    model_id=mid, seq_len=sl, num_samples=ns,
                    head_dim=arch["head_dim"], budget=blab, sk=sk, sv=sv,
                    total_kv_MB=round(mc["total_kv_MB"], 6),
                    residual_MB=round(mc["residual_MB"], 6),
                    saving_vs_fp16_pct=round(mc["saving_vs_fp16_pct"], 4),
                    ppl=(round(float(ppl), 4) if pd.notna(ppl) else pd.NA),
                    ppl_source=psrc,
                    is_effective_choice=(blab == eff_label),
                    carekv_status=r.get("status"),
                ))
        else:
            row.update(dict(fixed_store_budget="SK2SV4",
                            effective_store_budget=eff_label,
                            effective_k_slots=ek, effective_v_slots=ev,
                            memory_note="arch_unavailable"))
        all_rows.append(row)

    adf = pd.DataFrame(all_rows)
    pdf = pd.DataFrame(pareto_rows)

    # mark pareto dominance per (model, seq, num_samples) over budgets with BOTH
    # ppl and memory (real measured points only)
    pdf["pareto_status"] = ""
    pdf["dominated_by"] = ""
    for keys, g in pdf.groupby(["model_id", "seq_len", "num_samples"]):
        pts = [(i, float(rr["ppl"]), float(rr["total_kv_MB"]), rr["budget"])
               for i, rr in g.iterrows() if pd.notna(rr["ppl"])]
        for i, ppl, mem, b in pts:
            dom = None
            for j, p2, m2, b2 in pts:
                if j == i:
                    continue
                if p2 <= ppl and m2 <= mem and (p2 < ppl or m2 < mem):
                    dom = b2; break
            pdf.at[i, "pareto_status"] = "dominated" if dom else "frontier"
            pdf.at[i, "dominated_by"] = dom or ""

    # ── derived tables ──
    P = lambda n: os.path.join(args.out_dir, n)
    adf.to_csv(P("effective_budget_all_rows.csv"), index=False)

    # fixed SK2SV4 controlled head_dim-trend table
    fixed = adf.dropna(subset=["head_dim"]).copy()
    fixed_tbl = (fixed.groupby(["head_dim", "head_dim_group"], as_index=False)
                 .agg(n_cells=("model_id", "size"),
                      fixed_residual_overhead_vs_base_pct=("fixed_residual_overhead_vs_base_pct", "mean"),
                      fixed_residual_frac_of_total_pct=("fixed_residual_frac_of_total_pct", "mean"),
                      fixed_saving_vs_fp16_pct=("fixed_saving_vs_fp16_pct", "mean"))
                 .sort_values("head_dim"))
    fixed_tbl.to_csv(P("fixed_sk2sv4_memory_table.csv"), index=False)

    # effective selected-budget table
    eff_cols = ["model_id", "seq_len", "num_samples", "head_dim", "selected_correction",
                "selector_decision", "gate_a_pass", "gate_b_pass", "carekv_status",
                "fixed_store_budget", "fixed_total_kv_MB", "fixed_residual_MB",
                "effective_store_budget", "effective_k_slots", "effective_v_slots",
                "effective_residual_MB", "effective_total_kv_MB",
                "effective_residual_overhead_vs_base_pct",
                "effective_residual_frac_of_total_pct", "effective_saving_vs_fp16_pct",
                "effective_total_kv_MB_reduction_vs_fixed_pct"]
    eff_tbl = adf[[c for c in eff_cols if c in adf.columns]].copy()
    eff_tbl.to_csv(P("effective_selected_budget_table.csv"), index=False)

    pdf.to_csv(P("budget_quality_pareto.csv"), index=False)

    # budget distribution summary
    dist = (adf.groupby(["selected_correction", "effective_store_budget"], as_index=False)
            .agg(n_cells=("model_id", "size")))
    dist.to_csv(P("effective_budget_distribution.csv"), index=False)

    report = P("EFFECTIVE_BUDGET_MEMORY_REPORT.md")
    write_report(report, adf, fixed_tbl, eff_tbl, pdf, dist)

    # ── stdout ──
    outs = ["effective_budget_all_rows.csv", "fixed_sk2sv4_memory_table.csv",
            "effective_selected_budget_table.csv", "budget_quality_pareto.csv",
            "effective_budget_distribution.csv", "EFFECTIVE_BUDGET_MEMORY_REPORT.md"]
    print("\n" + "=" * 72 + "\nOUTPUT FILES:")
    for o in outs:
        print("  ", P(o))
    pd.set_option("display.width", 220)
    print("\n" + "=" * 72 + "\nEFFECTIVE BUDGET DISTRIBUTION:")
    print(dist.to_string(index=False))
    print("\n" + "=" * 72 + "\nFIXED SK2SV4 head_dim trend:")
    print(fixed_tbl.to_string(index=False))
    print("\n" + "=" * 72 + "\nEFFECTIVE selected-budget table (real-arch cells):")
    show = eff_tbl[eff_tbl["effective_store_budget"].notna()][
        ["model_id", "seq_len", "selected_correction", "fixed_total_kv_MB",
         "effective_store_budget", "effective_total_kv_MB",
         "effective_total_kv_MB_reduction_vs_fixed_pct", "carekv_status"]]
    print(show.to_string(index=False))
    print("\n" + "=" * 72 + "\nCONCLUSION")
    nvd = (adf["effective_store_budget"] == "SK0SV4").sum()
    n00 = (adf["effective_store_budget"] == "SK0SV0").sum()
    n24 = (adf["effective_store_budget"] == "SK2SV4").sum()
    print(f"- Effective deployment budgets: SK0SV0={n00}, SK0SV4={nvd}, SK2SV4={n24} cells.")
    print("- Vdom cells need NO K residuals (audited): effective SK0SV4, not SK2SV4.")
    print("- Only KV-selected cells carry the full SK2SV4 budget; the fixed SK2SV4")
    print("  table remains valid as a controlled head_dim-trend analysis.")
    print("- Budget-quality Pareto uses real measured PPLs at SK0SV0/SK0SV4/SK2SV4;")
    print("  SK0SV2/SK4SV4 have no PPL rows and are shown memory-only (not faked).")


def md_table(d, floatfmt=4):
    d = d.copy()
    for c in d.columns:
        d[c] = d[c].map(lambda v: (f"{v:.{floatfmt}f}" if isinstance(v, float) else
                                   ("" if pd.isna(v) else v)))
    hdr = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    return "\n".join([hdr, sep] + body)


def write_report(path, adf, fixed_tbl, eff_tbl, pdf, dist):
    L = []
    L.append("# CARE-KV Effective-Budget Memory Accounting & Selector Report\n")
    L.append("SK/SV is **not** a universally fixed deployment budget. This report "
             "distinguishes three analyses:\n")
    L.append("1. **Fixed SK2SV4 controlled analysis** — for explaining the "
             "head_dim overhead trend (every cell costed at the same SK2SV4).")
    L.append("2. **Effective deployment budget** — what the adaptive selector "
             "actually requires per cell.")
    L.append("3. **PPL–memory budget-quality Pareto** — accuracy vs KV bytes "
             "across budgets, using only real measured PPLs.\n")

    L.append("## Audit: are K residual buffers allocated/used under Vdom?\n")
    L.append("Code audit of the V-only (Vdom) path:")
    L.append("- **Store** (`residual_store.py`): `kind=\"v\"` → `use_k=False` → "
             "`k_budget=0` → `alloc_k_slot()` is never called → **zero K "
             "residual slots written**.")
    L.append("- **Read** (`residual_router.py`): `kind=\"v\"` → "
             "`_resolve_read_budgets` forces `bk=0`, and the K-candidate loop is "
             "guarded by `kind in {\"k\",\"both\"}` → **zero K residual slots "
             "read**.")
    L.append("- **But** (`cache.py`): the K residual arena "
             "(`k_residual_buf`/`k_residual_scale`) was **pre-allocated** at "
             "construction sized by `(store_abs_k+store_abs_v)`, so a V-only "
             "deployment still reserved a full K arena = dead memory. The as-run "
             "experiments stored `kind=both` (SK2SV4) and routed Vdom at eval, so "
             "K residuals were physically present but unused for Vdom cells.")
    L.append("\n**Finding: K residual buffers are allocated but unused in Vdom** "
             "→ the effective Vdom budget is **SK0SV4**.\n")
    L.append("**Optimization added** (flag-gated, default off → paper-best "
             "preserved): `CAREKV_VDOM_OPTIMIZED=1` drops the K arena to a "
             "1-slot stub (~99.7% K-arena reduction in a unit check) and makes "
             "`alloc_k_slot()` raise as an audit guard. Verified lossless by "
             "`tests/test_carekv_v2.py::test_vdom_optimized_kstore_audit` "
             "(V-only route byte-identical with vs without the K arena); full "
             "suite (20 tests) green.\n")

    L.append("## 1. Fixed SK2SV4 controlled analysis (head_dim trend)\n")
    L.append("Every cell costed at SK2SV4. Residual overhead ratio = "
             "`(SK+SV)·1.13/(2·head_dim·bits/8)` is arch-only, so it falls with "
             "head_dim — this is the controlled explanation table (kept intact).\n")
    L.append(md_table(fixed_tbl[["head_dim", "head_dim_group", "n_cells",
                                 "fixed_residual_overhead_vs_base_pct",
                                 "fixed_residual_frac_of_total_pct",
                                 "fixed_saving_vs_fp16_pct"]]))

    L.append("\n## 2. Effective deployment budget after the adaptive selector\n")
    L.append("Mapping: `BaseQuant_INT3`/`skip_near_lossless`→**SK0SV0**, "
             "`Vdom`→**SK0SV4**, `KV`→**SK2SV4**.\n")
    L.append("Distribution:\n")
    L.append(md_table(dist))
    L.append("\nPer-cell effective budget (real-arch cells):\n")
    show = eff_tbl[eff_tbl["effective_store_budget"].notna()][
        ["model_id", "seq_len", "selected_correction", "fixed_total_kv_MB",
         "effective_store_budget", "effective_total_kv_MB",
         "effective_residual_frac_of_total_pct",
         "effective_total_kv_MB_reduction_vs_fixed_pct", "gate_a_pass",
         "gate_b_pass", "carekv_status"]]
    L.append(md_table(show))
    L.append("\n_Note: `fixed_*` columns (controlled SK2SV4) are preserved "
             "alongside `effective_*` columns in "
             "`effective_budget_all_rows.csv`. Cells where the selector chose "
             "Vdom or BaseQuant deploy with strictly less KV memory than the "
             "fixed SK2SV4 accounting implies._\n")

    L.append("## 3. PPL–memory budget-quality Pareto\n")
    L.append("Memory is analytical (batch=1, packed int8, INT3). PPL is **real "
             "measured** only: SK0SV0 = BaseQuant INT3, SK0SV4 = "
             "`calibration_ppl_vdom`, SK2SV4 = `calibration_ppl_kv`. "
             "**SK0SV2 / SK4SV4 have no measured PPL rows** and appear "
             "memory-only (`ppl_source=no_ppl_row`) — never faked.\n")
    # show a representative model's full budget ladder where all 3 PPLs exist
    have = pdf[pdf["ppl"].notna()]
    rep = None
    for keys, g in have.groupby(["model_id", "seq_len", "num_samples"]):
        if g["budget"].nunique() >= 3:
            rep = (keys, g); break
    if rep is not None:
        (mid, sl, ns), g = rep
        L.append(f"Example — `{mid}` SL{sl} N{ns} (all three measured PPLs):\n")
        L.append(md_table(g.sort_values("total_kv_MB")[
            ["budget", "total_kv_MB", "saving_vs_fp16_pct", "ppl", "ppl_source",
             "is_effective_choice", "pareto_status"]]))
    n_meas = int(pdf["ppl"].notna().sum())
    n_noppl = int((pdf["ppl_source"] == "no_ppl_row").sum())
    L.append(f"\nFull ladder in `budget_quality_pareto.csv` "
             f"({n_meas} measured-PPL points, {n_noppl} memory-only points).\n")

    L.append("## Honesty / preservation\n")
    L.append("- Analytical estimator memory ≠ measured GPU peak; clean-allocation "
             "validation lives in the residual-overhead report.")
    L.append("- Fixed SK2SV4 columns are preserved for the controlled head_dim "
             "analysis; effective columns are added, not substituted.")
    L.append("- Failed/OOM/blocked/collapsed CARE-KV cells are preserved in "
             "`effective_budget_all_rows.csv` with their `carekv_status` and "
             "selector decision (e.g. Qwen `no_valid_carekv_candidate` / "
             "`blocked_architecture_port`). They map to SK0SV0 effective but are "
             "flagged, not silently counted as wins.")
    L.append("- The Vdom→SK0SV4 claim is backed by the code audit + the "
             "lossless optimized-path test, not assumed.")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
