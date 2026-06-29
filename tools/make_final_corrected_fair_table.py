"""tools/make_final_corrected_fair_table.py

Final corrected, bit-width-FAIR quality comparison table.

Fair comparison is INT3-ONLY among {BaseQuant_INT3, Adaptive_CAREKV_INT3,
TurboQuant_INT3_standalone}; tie threshold 0.02 PPL. INT4 is higher-bit
reference only (never in the fair table). INT2 is excluded from the main table
and recorded as unstable_outlier_collapse / paper_usable=no in the failure
table. TurboQuant_plus_CAREKV stays unsupported and is preserved in the
failure/unsupported table.

Inputs: results/score_aware_chunked_full/score_aware_chunked_7b.csv (quality run).
Outputs (results/final_corrected_fair_table/):
  final_quality_main_table.csv
  final_quality_appendix_all_rows.csv
  final_quality_failure_or_unsupported.csv
  FINAL_QUALITY_FAIR_COMPARISON_REPORT.md
Append-safe (writes its own files; never touches the source CSV). Not committed.
"""
import csv, os, math
from collections import Counter

SRC = "results/score_aware_chunked_full/score_aware_chunked_7b.csv"
D = "results/final_corrected_fair_table"
INT3_FAIR = ["BaseQuant_INT3", "Adaptive_CAREKV_INT3", "TurboQuant_INT3_standalone"]
INT4_REF = ["BaseQuant_INT4", "TurboQuant_INT4_standalone"]
TIE = 0.02


def fv(x):
    try: v = float(x); return v if math.isfinite(v) else None
    except Exception: return None
def short(m): return str(m).split("/")[-1]


def effective_budget(selected, decision, status):
    """Map CARE-KV selection → effective store budget tag."""
    if status == "blocked_oom" or status.startswith("blocked"):
        return "no_valid_carekv"
    if decision == "skip_near_lossless" or selected in ("BaseQuant_INT3", "skip", ""):
        return "SK0SV0"
    if selected == "Vdom":
        return "SK0SV4"
    if selected in ("KV", "K+V"):
        return "SK2SV4"
    if selected in ("none",):
        return "no_valid_carekv"
    return "SK0SV0"


def main():
    if not os.path.exists(SRC):
        print(f"source CSV missing: {SRC}"); return
    os.makedirs(D, exist_ok=True)
    rows = list(csv.DictReader(open(SRC)))
    src_cols = list(rows[0].keys())
    keys = sorted(set((r["model_id"], r["seq_len"]) for r in rows if r.get("method") not in ("ALL", "")),
                  key=lambda x: (short(x[0]), int(x[1]) if str(x[1]).isdigit() else 0))

    main_rows, fail_rows = [], []
    tally = Counter()
    for mid, SL in keys:
        g = {r["method"]: r for r in rows if r["model_id"] == mid and r["seq_len"] == SL}
        fp = fv(g.get("fp16", {}).get("ppl"))
        ck = g.get("Adaptive_CAREKV_INT3", {})
        sel = ck.get("selected_correction", ""); dec = ck.get("selector_decision", ""); ck_st = ck.get("status", "")
        b3 = fv(g.get("BaseQuant_INT3", {}).get("ppl"))
        ckv = fv(ck.get("ppl")); tq3 = fv(g.get("TurboQuant_INT3_standalone", {}).get("ppl"))
        # fair candidates: real + finite + not collapsed (<5x fp16)
        cands = []
        for m in INT3_FAIR:
            r = g.get(m); v = fv(r.get("ppl")) if r else None
            if v is not None and r.get("status") in ("real",) and (fp is None or v < 5 * fp):
                cands.append((v, m))
        cands.sort()
        if not cands:
            fair_result, tied = "no_valid_int3", ""
        else:
            best_v, best_m = cands[0]
            tied = [m for v, m in cands if abs(v - best_v) <= TIE]
            if len(tied) >= 2:
                fair_result = "tie"
            else:
                fair_result = best_m
        tally[fair_result if fair_result in ("tie", "no_valid_int3") else fair_result] += 1
        notes = []
        if dec == "skip_near_lossless": notes.append("base near-lossless; CARE-KV skipped (no over-claim)")
        if ck.get("gate_b_pass") == "False": notes.append("CARE-KV Gate B FAIL → fell back to base")
        if ck.get("gate_a_pass") == "False": notes.append("CARE-KV Gate A FAIL")
        main_rows.append(dict(
            model_id=short(mid), seq_len=SL,
            fp16_ppl=g.get("fp16", {}).get("ppl", ""),
            basequant_int3_ppl=g.get("BaseQuant_INT3", {}).get("ppl", ""),
            carekv_int3_ppl=ck.get("ppl", ""),
            carekv_selected_correction=sel or "-",
            carekv_effective_budget=effective_budget(sel, dec, ck_st),
            turboquant_int3_ppl=g.get("TurboQuant_INT3_standalone", {}).get("ppl", ""),
            fair_int3_result=fair_result,
            # Full, unambiguous method names. NEVER abbreviate to "TurboQuant+CARE-KV"
            # (that reads like the unsupported TurboQuant_plus_CAREKV). Ties are
            # always between two of {BaseQuant_INT3, Adaptive_CAREKV_INT3,
            # TurboQuant_INT3_standalone}.
            fair_int3_tied_methods=(" and ".join(tied) if fair_result == "tie" else ""),
            notes="; ".join(notes)))
        # failure/unsupported rows for this (model,SL): INT2 + TurboQuant+CARE-KV + blocked CARE-KV.
        # INT2 is NEVER paper-usable and is excluded from the fair INT3 comparison; even the
        # "borderline" OpenLLaMA INT2 (~38-42, ~4-5x fp16) is extreme degradation, so ALL INT2 rows
        # are uniformly marked unstable_outlier_collapse / paper_usable=no.
        for m in ("TurboQuant_INT2_standalone",):
            r = g.get(m)
            if r:
                fail_rows.append(dict(model_id=short(mid), seq_len=SL, method=m, ppl=r.get("ppl", ""),
                                      status="unstable_outlier_collapse", paper_usable="no",
                                      reason="INT2 collapse / extreme PPL degradation"))
        tqck = g.get("TurboQuant_plus_CAREKV")
        if tqck:
            fail_rows.append(dict(model_id=short(mid), seq_len=SL, method="TurboQuant_plus_CAREKV", ppl="",
                                  status="unsupported", paper_usable="no",
                                  reason="QJL is a score-level inner-product estimator, while CARE-KV corrects "
                                         "reconstructed K/V values; direct stacking would redefine the methods."))
        if ck_st.startswith("blocked"):
            fail_rows.append(dict(model_id=short(mid), seq_len=SL, method="Adaptive_CAREKV_INT3", ppl=ck.get("ppl", ""),
                                  status=ck_st, paper_usable="no", reason=ck.get("blocker", "")))

    # write main table
    mcols = ["model_id", "seq_len", "fp16_ppl", "basequant_int3_ppl", "carekv_int3_ppl",
             "carekv_selected_correction", "carekv_effective_budget", "turboquant_int3_ppl",
             "fair_int3_result", "fair_int3_tied_methods", "notes"]
    with open(f"{D}/final_quality_main_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mcols); w.writeheader()
        for r in main_rows: w.writerow(r)
    # appendix: every source row (kept intact)
    with open(f"{D}/final_quality_appendix_all_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=src_cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow(r)
    # failure / unsupported
    fcols = ["model_id", "seq_len", "method", "ppl", "status", "paper_usable", "reason"]
    with open(f"{D}/final_quality_failure_or_unsupported.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fcols); w.writeheader()
        for r in fail_rows: w.writerow(r)

    # ---- report ----
    o = ["# FINAL quality fair comparison (INT3-only, bit-width-fair)\n",
         "> Fair comparison is **INT3-only** among BaseQuant_INT3 / Adaptive_CAREKV_INT3 / "
         "TurboQuant_INT3_standalone; **tie ≤ 0.02 PPL**. INT4 is higher-bit reference only (not in "
         "this table). INT2 is excluded from the main table (recorded `unstable_outlier_collapse`, "
         "`paper_usable=no` in the failure table). TurboQuant+CARE-KV stays `unsupported`. "
         "Source: score-aware chunked deterministic run (chunked cs=128, TF32-off, attention_output "
         "V-score). All OOM/failed/unsupported rows preserved.\n",
         "## Main quality table (fair INT3)\n",
         "| model | SL | fp16 | BaseQ INT3 | CARE-KV INT3 (sel / budget) | TurboQ INT3 | **fair INT3 result** | tied methods | notes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in main_rows:
        o.append(f"| {r['model_id']} | {r['seq_len']} | {r['fp16_ppl']} | {r['basequant_int3_ppl']} "
                 f"| {r['carekv_int3_ppl']} ({r['carekv_selected_correction']} / {r['carekv_effective_budget']}) "
                 f"| {r['turboquant_int3_ppl']} | **{r['fair_int3_result']}** | {r['fair_int3_tied_methods']} | {r['notes']} |")
    o.append(f"\n**Fair INT3 tally:** " + ", ".join(f"{k}={v}" for k, v in tally.most_common()) + ".\n")
    o.append("## Failure / collapse / unsupported (preserved)\n")
    o.append("| model | SL | method | ppl | status | paper_usable | reason |")
    o.append("|---|---|---|---|---|---|---|")
    for r in fail_rows:
        o.append(f"| {r['model_id']} | {r['seq_len']} | {r['method']} | {r['ppl'] or '—'} | {r['status']} "
                 f"| {r['paper_usable']} | {r['reason'][:60]} |")
    o.append("\nINT4 rows are in the appendix (`final_quality_appendix_all_rows.csv`) as higher-bit reference.\n")
    open(f"{D}/FINAL_QUALITY_FAIR_COMPARISON_REPORT.md", "w").write("\n".join(o) + "\n")
    print(f"main rows: {len(main_rows)}; failure/unsupported rows: {len(fail_rows)}")
    print("fair INT3 tally:", dict(tally))
    print("wrote 3 CSVs + FINAL_QUALITY_FAIR_COMPARISON_REPORT.md to", D)


def short_m(m):
    return {"BaseQuant_INT3": "BaseQuant", "Adaptive_CAREKV_INT3": "CARE-KV",
            "TurboQuant_INT3_standalone": "TurboQuant"}.get(m, m)


if __name__ == "__main__":
    main()
