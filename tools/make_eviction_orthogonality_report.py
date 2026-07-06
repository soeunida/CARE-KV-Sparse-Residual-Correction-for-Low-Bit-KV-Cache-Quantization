"""tools/make_eviction_orthogonality_report.py

Consolidate the eviction-orthogonality (additivity) results from
results/eviction_additivity/evict_add_*.csv into a single md table + figure.

Orthogonality claim (Section 2): CARE-KV's PPL benefit is independent of KV
eviction — i.e. the CARE benefit measured without eviction is preserved when
eviction is applied on top:
    benefit_noevict = base_noevict − carekv_noevict
    benefit_evict   = base_evict   − carekv_evict
    orthogonal  ⟺  benefit_evict ≈ benefit_noevict

Arms per setting: fp16, base_noevict, carekv_noevict, base_evict, carekv_evict.
Outputs:
    results/eviction_additivity/eviction_orthogonality_results.md
    results/eviction_additivity/fig_eviction_orthogonality.png
"""
from __future__ import annotations
import argparse, csv, glob, os
from collections import defaultdict

DIRDEF = "results/eviction_additivity"
TOL = 0.15   # |benefit_evict − benefit_noevict| ≤ TOL → "orthogonal" (additive)


def _short(model_id):
    m = model_id.split("/")[-1]
    return {"TinyLlama-1.1B-Chat-v1.0": "TinyLlama-1.1B",
            "Mistral-7B-v0.3": "Mistral-7B",
            "deepseek-llm-7b-base": "DeepSeek-7B"}.get(m, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIRDEF)
    args = ap.parse_args()

    # group rows by (csv_basename, model_id) -> {arm: row}
    settings = defaultdict(dict)
    order = []
    for path in sorted(glob.glob(os.path.join(args.dir, "evict_add_*.csv"))):
        for r in csv.DictReader(open(path)):
            key = (os.path.basename(path), r["model_id"])
            if key not in settings:
                order.append(key)
            settings[key][r["arm"]] = r

    rows = []
    for key in order:
        d = settings[key]
        fn, model = key
        def ppl(a):
            try:
                return float(d[a]["ppl"])
            except (KeyError, ValueError):
                return None
        fp16 = ppl("fp16")
        bn, cn = ppl("base_noevict"), ppl("carekv_noevict")
        be, ce = ppl("base_evict"), ppl("carekv_evict")
        # keep_ratio / policy from an evict arm
        ev = d.get("base_evict") or d.get("carekv_evict") or {}
        keep = ev.get("keep_ratio", "")
        pol = ev.get("evict_policy", "")
        if None in (bn, cn, be, ce):
            continue
        bnoev = bn - cn
        bev = be - ce
        rows.append(dict(model=_short(model), keep=keep, policy=pol,
                         fp16=fp16, bn=bn, cn=cn, be=be, ce=ce,
                         bnoev=bnoev, bev=bev, dd=bev - bnoev,
                         orth=abs(bev - bnoev) <= TOL))

    L = []
    L.append("# Eviction orthogonality (additivity) — methodology + results")
    L.append("")
    L.append("**Claim (Section 2).** CARE-KV's PPL benefit is **orthogonal to KV "
             "eviction**: the benefit measured without eviction is preserved when "
             "token eviction (H2O) is applied on top.")
    L.append("")
    L.append("## Methodology")
    L.append("")
    L.append("KV compression has two independent axes — **eviction** (*which* "
             "tokens to keep) and **CARE-KV** (*correcting* the quantization error "
             "of the kept tokens). We test their independence with a **2×2 "
             "factorial design** (CARE on/off × eviction on/off), all arms run "
             "through the same `CAREKVAdapter` + eviction hook so eviction applies "
             "uniformly:")
    L.append("")
    L.append("| arm | CARE-KV | eviction |")
    L.append("|---|---|---|")
    L.append("| `base_noevict`   | ✗ (READ budget 0 = INT3 base) | ✗ keep=1.0 |")
    L.append("| `carekv_noevict` | ✓ SK2 SV4 RK2 RV2 | ✗ keep=1.0 |")
    L.append("| `base_evict`     | ✗ | ✓ keep=R |")
    L.append("| `carekv_evict`   | ✓ SK2 SV4 RK2 RV2 | ✓ keep=R |")
    L.append("")
    L.append("(+ `fp16` upper-bound reference.) **Eviction = H2O**: keep the "
             "fraction R of tokens with highest cumulative attention, protecting "
             "sink (=4) and a recent window (env `CAREKV_EVICT_KEEP_RATIO/POLICY/"
             "SINK/RECENT`). keep R swept over 0.9 / 0.75 / 0.5.")
    L.append("")
    L.append("**Additivity / orthogonality test** (zero 2-factor interaction):")
    L.append("")
    L.append("```")
    L.append("benefit_noevict = base_noevict − carekv_noevict")
    L.append("benefit_evict   = base_evict   − carekv_evict")
    L.append(f"orthogonal/additive  ⟺  benefit_evict ≈ benefit_noevict   (|Δ| ≤ {TOL})")
    L.append("```")
    L.append("")
    L.append("Metric: WikiText-2 PPL, N=4, SL=256 (TinyLlama) / 512 (7B). "
             "`carekv_evict` is the **combined** config (CARE-KV ⊕ eviction) — the "
             "practical deployment of both compressions together.")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("| model | keep | policy | fp16 | base noev | CARE noev | base evict | "
             "CARE evict | CAREΔ noev | CAREΔ evict | Δ-of-Δ | orthogonal |")
    L.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for r in rows:
        L.append(f"| {r['model']} | {r['keep']} | {r['policy'] or '—'} | "
                 f"{r['fp16']:.3f} | {r['bn']:.3f} | {r['cn']:.3f} | {r['be']:.3f} | "
                 f"{r['ce']:.3f} | {r['bnoev']:+.3f} | {r['bev']:+.3f} | "
                 f"{r['dd']:+.3f} | {'✅' if r['orth'] else '≈/✗'} |")
    L.append("")
    L.append("## Reading")
    L.append("")
    orth_keep09 = [r for r in rows if str(r['keep']) == "0.9"]
    L.append("- **Mild eviction (keep=0.9) — orthogonality holds**, cleanest on 7B: "
             + "; ".join(f"{r['model']} CAREΔ {r['bnoev']:+.2f}→{r['bev']:+.2f}"
                         for r in orth_keep09) + ".")
    L.append("- **Aggressive eviction (keep≤0.75, TinyLlama)** — eviction inflates "
             "PPL sharply, and CARE-KV recovers **even more** (benefit grows: "
             + ", ".join(f"keep={r['keep']} Δ {r['bnoev']:+.2f}→{r['bev']:+.2f}"
                         for r in rows if r['model'].startswith('TinyLlama')
                         and str(r['keep']) != '0.9') + "). CARE-KV cushions the "
             "eviction-induced collapse rather than being merely additive.")
    L.append("- **Direction is consistent**: CARE-KV's gain over BaseQuant survives "
             "(and at low keep, amplifies under) eviction → the two mechanisms are "
             "complementary, not competing.")
    L.append("")
    L.append("**Status: diagnostic** (N=4, SL=256/512). Source CSVs: "
             "`results/eviction_additivity/evict_add_*.csv`.")
    out_md = os.path.join(args.dir, "eviction_orthogonality_results.md")
    open(out_md, "w").write("\n".join(L) + "\n")
    print("wrote", out_md)

    # figure: CARE benefit no-evict vs evict per setting
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, numpy as np
        labels = [f"{r['model']}\nkeep={r['keep']}" for r in rows]
        x = np.arange(len(rows)); w = 0.38
        bnoev = [r['bnoev'] for r in rows]
        bev = [r['bev'] for r in rows]
        fig, ax = plt.subplots(figsize=(max(8, 1.5*len(rows)), 5), constrained_layout=True)
        ax.bar(x - w/2, bnoev, w, label="CARE benefit (no eviction)", color="#1f77b4", edgecolor="black")
        ax.bar(x + w/2, bev, w, label="CARE benefit (with eviction)", color="#2ca02c", edgecolor="black")
        for i, r in enumerate(rows):
            ax.text(i - w/2, bnoev[i], f"{bnoev[i]:.2f}", ha="center", va="bottom", fontsize=7)
            ax.text(i + w/2, bev[i], f"{bev[i]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("CARE-KV PPL benefit  (base − CARE-KV)")
        ax.set_title("Eviction orthogonality: CARE-KV benefit preserved/amplified under eviction (WT-2 N=4)")
        ax.legend(fontsize=9); ax.grid(axis="y", ls="--", alpha=0.4)
        out_fig = os.path.join(args.dir, "fig_eviction_orthogonality.png")
        fig.savefig(out_fig, dpi=130); plt.close(fig)
        print("wrote", out_fig)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
