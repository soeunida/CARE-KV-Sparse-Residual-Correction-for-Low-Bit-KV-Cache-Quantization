"""tools/make_phase11c_summary.py — regenerate the Phase 11C summary from an EXISTING
phase11c_all_rows.csv, applying the finite-collapse paper_usable guard post-hoc (no
model re-evaluation). Mirrors the in-driver guard:

  finite_collapse(ppl, fp16) := finite(ppl) & finite(fp16) & ppl > max(100, 5*fp16)

  - BaseQuant collapse  → base-collapsed setting; derived CARE-KV/Turbo not usable.
  - CARE-KV collapse (base ok)  → paper_usable=no, reason carekv_finite_collapse.
  - Turbo collapse / base-collapsed → paper_usable=no, reason turbo_finite_collapse.

Writes phase11c_summary_corrected.csv next to the input and prints the table.

Usage: python tools/make_phase11c_summary.py [results/phase11c_cached_models/phase11c_all_rows.csv]
"""
import sys, csv, math, os

IN = sys.argv[1] if len(sys.argv) > 1 else "results/phase11c_cached_models/phase11c_all_rows.csv"


def fin(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def fval(x):
    try:
        return float(x)
    except Exception:
        return None


def finite_collapse(ppl, fp):
    return (fin(ppl) and fin(fp) and float(ppl) > max(100.0, 5.0 * float(fp)))


def main():
    rows = list(csv.DictReader(open(IN)))
    # group by (model_id, seq_len)
    models, fam = [], {}
    for r in rows:
        key = r["model_id"]
        if key not in models and r.get("method") not in ("ALL",) and r.get("status") != "skipped_not_cached":
            models.append(key)
        fam[key] = r.get("family", "?")

    def get(mid, meth, col):
        for r in rows:
            if r["model_id"] == mid and r["method"] == meth:
                return r.get(col, "")
        return ""

    out_cols = ["model_id", "family", "seq_len", "num_samples", "fp16", "BaseQuant_INT3",
                "CAREKV_current_SK2SV4", "CAREKV_combined_kvscore", "TurboQuant_INT3",
                "winner", "delta_vs_basequant", "delta_vs_turboquant", "K_reads", "V_reads",
                "correction_type", "k_correction_active", "paper_usable", "paper_usable_reason"]
    summary = []
    hdr = (f"{'model':34s}{'fam':14s}{'fp16':>8s}{'Base':>9s}{'CKV':>10s}{'Turbo':>9s}"
           f"{'dBase':>9s}{'dTurbo':>8s}{'type':>11s}{'kact':>6s}{'usable':>7s}  reason / winner")
    print(hdr); print("-" * len(hdr))
    n_usable = 0
    for mid in models:
        sl = get(mid, "fp16", "seq_len") or get(mid, "CAREKV_current_SK2SV4", "seq_len")
        ns = get(mid, "fp16", "num_samples")
        fp = fval(get(mid, "fp16", "ppl"))
        bq = fval(get(mid, "BaseQuant_INT3", "ppl"))
        ck = fval(get(mid, "CAREKV_current_SK2SV4", "ppl"))
        cmb = fval(get(mid, "CAREKV_combined_kvscore", "ppl"))
        tq = fval(get(mid, "TurboQuant_INT3_standalone", "ppl"))
        K = get(mid, "CAREKV_current_SK2SV4", "K_reads")
        V = get(mid, "CAREKV_current_SK2SV4", "V_reads")
        ctype = get(mid, "CAREKV_current_SK2SV4", "correction_type")
        kact = get(mid, "CAREKV_current_SK2SV4", "k_correction_active")
        dK = fval(get(mid, "CAREKV_current_SK2SV4", "correction_delta_norm_K"))
        dV = fval(get(mid, "CAREKV_current_SK2SV4", "correction_delta_norm_V"))

        base_collapsed = finite_collapse(bq, fp)
        ck_collapse = finite_collapse(ck, fp)
        # corrected CARE-KV paper_usable
        if base_collapsed:
            usable, reason = "no", "base_collapsed_setting"
        elif ck is None:
            usable, reason = "no", "non-finite / blocked"
        elif ck_collapse:
            usable, reason = "no", ("carekv_finite_collapse (k_correction_blowup)"
                                    if (dK is not None and dV is not None and dK > dV)
                                    else "carekv_finite_collapse")
        else:
            usable, reason = "yes", ""
        if usable == "yes":
            n_usable += 1

        # winner among non-collapsed quant methods (lower PPL better)
        cand = {}
        if bq is not None and not base_collapsed:
            cand["Base"] = bq
        if tq is not None and not finite_collapse(tq, fp) and not base_collapsed:
            cand["Turbo"] = tq
        if ck is not None and not ck_collapse and not base_collapsed:
            cand["CARE-KV"] = ck
        winner = min(cand, key=cand.get) if cand else "none (all collapse/blocked)"
        dBase = round(ck - bq, 4) if (ck is not None and bq is not None) else ""
        dTurbo = round(ck - tq, 4) if (ck is not None and tq is not None) else ""

        summary.append(dict(model_id=mid, family=fam[mid], seq_len=sl, num_samples=ns,
                            fp16=fp, BaseQuant_INT3=bq, CAREKV_current_SK2SV4=ck,
                            CAREKV_combined_kvscore=(cmb if cmb is not None else ""),
                            TurboQuant_INT3=(tq if tq is not None else "n/a"),
                            winner=winner, delta_vs_basequant=dBase, delta_vs_turboquant=dTurbo,
                            K_reads=K, V_reads=V, correction_type=ctype,
                            k_correction_active=kact, paper_usable=usable,
                            paper_usable_reason=reason))
        tag = f"{reason or 'win=' + winner}"
        print(f"{mid:34s}{fam[mid]:14s}{(f'{fp:.2f}' if fp else '-'):>8s}"
              f"{(f'{bq:.2f}' if bq else '-'):>9s}{(f'{ck:.2f}' if ck else '-'):>10s}"
              f"{(f'{tq:.2f}' if tq else 'n/a'):>9s}{str(dBase):>9s}{str(dTurbo):>8s}"
              f"{ctype:>11s}{str(kact):>6s}{usable:>7s}  {tag}")

    skipped = [r["model_id"] for r in rows if r.get("status") == "skipped_not_cached"]
    out = os.path.join(os.path.dirname(IN), "phase11c_summary_corrected.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols); w.writeheader()
        for s in summary:
            w.writerow(s)
    print(f"\nstable paper-usable models: {n_usable}")
    print(f"skipped (uncached): {skipped}")
    print(f"corrected summary written: {out}")


if __name__ == "__main__":
    main()
