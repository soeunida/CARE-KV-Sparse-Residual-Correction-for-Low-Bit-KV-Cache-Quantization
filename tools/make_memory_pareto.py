"""tools/make_memory_pareto.py — PPL vs KV-memory Pareto analysis (no GPU).

Positions CARE-KV on a quality/memory Pareto front vs BaseQuant / TurboQuant / fp16.
PPL from the GQA scaling run (results/gqa_scaling/gqa_scaling.csv). KV memory as a
fraction of fp16 from the validated analytic accounting (kv_residual_memory_scaling.csv,
INT3, SK2SV4): BaseQuant ≈ 0.203×fp16, CARE-KV ≈ 0.230×fp16 (base +~13% residual),
TurboQuant ≈ base (QJL qjl_m=0 stores no residual). fp16 = 1.0.

A point (mem, ppl) is Pareto-DOMINATED if another method has ≤ mem AND ≤ ppl.

Writes results/memory_pareto/pareto.csv + a summary printed to stdout.
"""
import csv, os, math

GQA = "results/gqa_scaling/gqa_scaling.csv"
OUT = "results/memory_pareto/pareto.csv"
# KV memory as fraction of fp16 (INT3, SK2SV4) — structural, from kv_residual_memory_scaling.csv
MEM_FRAC = {"fp16": 1.0, "base_int3": 0.203, "carekv_SK2SV4": 0.230, "turbo_int3": 0.203}
LABEL = {"fp16": "fp16", "base_int3": "BaseQuant_INT3", "carekv_SK2SV4": "CARE-KV_SK2SV4",
         "turbo_int3": "TurboQuant_INT3"}


def fv(x):
    try: return float(x)
    except Exception: return None


def main():
    rows = list(csv.DictReader(open(GQA)))
    models = []
    for r in rows:
        if r["model_id"] not in models:
            models.append(r["model_id"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out_rows = []
    print(f"{'model':16s}{'method':17s}{'KV_MB':>9s}{'mem×fp16':>9s}{'ppl':>8s}{'pareto':>9s}")
    dom_by_turbo = 0
    for mid in models:
        def g(arm, col="ppl"):
            for r in rows:
                if r["model_id"] == mid and r["arm"] == arm:
                    return r.get(col, "")
            return ""
        L = int(g("fp16", "num_layers")); Hkv = int(g("fp16", "num_kv_heads"))
        hd = int(g("fp16", "head_dim")); SL = int(g("fp16", "seq_len") or 512)
        fp16_MB = 2 * L * Hkv * SL * hd * 2 / 1e6      # K+V, fp16 (2 bytes)
        pts = []
        for arm in ["fp16", "base_int3", "carekv_SK2SV4", "turbo_int3"]:
            ppl = fv(g(arm))
            if ppl is None:
                continue
            mem = round(fp16_MB * MEM_FRAC[arm], 3)
            pts.append(dict(arm=arm, mem=mem, ppl=ppl, mem_frac=MEM_FRAC[arm]))
        # Pareto domination within this model
        for p in pts:
            dominated_by = [LABEL[q["arm"]] for q in pts
                            if q is not p and q["mem"] <= p["mem"] and q["ppl"] <= p["ppl"]
                            and (q["mem"] < p["mem"] or q["ppl"] < p["ppl"])]
            p["pareto"] = "dominated" if dominated_by else "on-front"
            p["dominated_by"] = ";".join(dominated_by)
            if p["arm"] == "carekv_SK2SV4" and "TurboQuant_INT3" in dominated_by:
                dom_by_turbo += 1
        for p in sorted(pts, key=lambda x: x["mem"]):
            print(f"{mid.split('/')[-1][:15]:16s}{LABEL[p['arm']]:17s}{p['mem']:>9.1f}"
                  f"{p['mem_frac']:>9.3f}{p['ppl']:>8.2f}{p['pareto']:>9s}"
                  + (f"  <-{p['dominated_by']}" if p['dominated_by'] else ""))
            out_rows.append(dict(model_id=mid, method=LABEL[p["arm"]], kv_MB=p["mem"],
                                 mem_frac_fp16=p["mem_frac"], ppl=p["ppl"], pareto=p["pareto"],
                                 dominated_by=p["dominated_by"], seq_len=SL))
        print()
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model_id", "method", "kv_MB", "mem_frac_fp16",
                                          "ppl", "pareto", "dominated_by", "seq_len"])
        w.writeheader()
        for r in out_rows: w.writerow(r)
    print(f"CARE-KV dominated by TurboQuant (mem AND ppl) in {dom_by_turbo}/{len(models)} models (GQA N=4).")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
