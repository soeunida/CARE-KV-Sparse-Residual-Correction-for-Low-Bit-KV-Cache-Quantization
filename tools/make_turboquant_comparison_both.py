"""tools/make_turboquant_comparison_both.py

CARE-KV vs TurboQuant (fair INT3) — TWO verdict views from the same audited PPLs
(results/final_corrected_fair_table/final_quality_main_table.csv):

  View 1 (margin-0.02): ±0.02 PPL tie band on ΔCARE−Turbo over the 11 settings
          where CARE-KV produced an active correction (DeepSeek SL1024, where
          CARE-KV fell back to BaseQuant, is n/a). → 3 win / 2 tie / 6 loss.
  View 2 (audited):     the table's own `fair_int3_result` over all 12 settings.
          → 2 win / 3 tie / 7 loss.

Outputs:
  results/final_corrected_fair_table/turboquant_comparison_both.md
  results/final_corrected_fair_table/fig_turboquant_margin002.png
  results/final_corrected_fair_table/fig_turboquant_audited.png
"""
from __future__ import annotations
import argparse, csv, os

SRC = "results/final_corrected_fair_table/final_quality_main_table.csv"
ODIR = "results/final_corrected_fair_table"
TOL = 0.02
SHORT = {"mistralai/Mistral-7B-v0.3": "Mistral-7B", "01-ai/Yi-6B": "Yi-6B",
         "deepseek-ai/deepseek-llm-7b-base": "DeepSeek-7B",
         "openlm-research/open_llama_7b_v2": "OpenLLaMA-7B"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--odir", default=ODIR)
    args = ap.parse_args()
    raw = list(csv.DictReader(open(args.src)))

    R = []
    for r in raw:
        m = SHORT.get(r["model_id"], r["model_id"].split("/")[-1])
        sl = int(r["seq_len"])
        b = _f(r["basequant_int3_ppl"]); c = _f(r["carekv_int3_ppl"])
        tq = _f(r["turboquant_int3_ppl"])
        d = (c - tq) if (c is not None and tq is not None) else None
        # margin-0.02 view: n/a only where CARE-KV had no valid result (Gate
        # FAIL → no_valid_carekv, e.g. DeepSeek SL1024). A deliberate near-
        # lossless skip (SK0SV0, e.g. Yi-6B SL256) IS applicable.
        budget = (r.get("carekv_effective_budget") or "").strip().lower()
        na = (d is None) or ("no_valid" in budget)
        if na:
            v1 = "n/a"
        elif d < -TOL:
            v1 = "win"
        elif d > TOL:
            v1 = "loss"
        else:
            v1 = "tie"
        # audited view
        fr = (r.get("fair_int3_result") or "").strip()
        v2 = "win" if ("CARE" in fr) else ("loss" if "TurboQuant" in fr else "tie")
        R.append(dict(m=m, sl=sl, b=b, c=c, tq=tq, d=d, v1=v1, v2=v2))

    def tally(key):
        w = sum(1 for r in R if r[key] == "win")
        t = sum(1 for r in R if r[key] == "tie")
        l = sum(1 for r in R if r[key] == "loss")
        na = sum(1 for r in R if r[key] == "n/a")
        return w, t, l, na
    w1, t1, l1, na1 = tally("v1")
    w2, t2, l2, na2 = tally("v2")
    SYM = {"win": "✅", "tie": "≈", "loss": "✗", "n/a": "—"}

    L = ["# CARE-KV vs TurboQuant — fair INT3 (two verdict views)", "",
         "WikiText-2 PPL, fixed INT3. Same audited PPLs, two classifications.",
         f"Source: `{os.path.basename(args.src)}`.", "",
         f"- **View 1 (margin ±{TOL}, 11 active settings):** "
         f"**{w1} win / {t1} tie / {l1} loss** ({na1} n/a).",
         f"- **View 2 (audited `fair_int3_result`, 12 settings):** "
         f"**{w2} win / {t2} tie / {l2} loss**.", "",
         "| Model | Seq | BaseQuant | CARE-KV | TurboQuant | ΔCARE−Turbo | "
         "View1 (±0.02) | View2 (audited) |",
         "|---|---:|---:|---:|---:|---:|:--:|:--:|"]
    for r in R:
        L.append(f"| {r['m']} | {r['sl']} | {r['b']:.3f} | **{r['c']:.3f}** | "
                 f"{r['tq']:.3f} | {('%+.3f'%r['d']) if r['d'] is not None else '—'} | "
                 f"{SYM[r['v1']]} {r['v1']} | {SYM[r['v2']]} {r['v2']} |")
    L.append("")
    for tag, key, (w, t, l, na) in (("View 1 (margin ±0.02)", "v1", (w1, t1, l1, na1)),
                                    ("View 2 (audited)", "v2", (w2, t2, l2, na2))):
        L.append(f"## {tag} — CARE-KV wins ({w})")
        L.append("")
        L.append("| Model | Seq | CARE-KV | TurboQuant | Δ |")
        L.append("|---|---:|---:|---:|---:|")
        for r in sorted([x for x in R if x[key] == "win"], key=lambda x: x["d"]):
            L.append(f"| {r['m']} | {r['sl']} | **{r['c']:.3f}** | {r['tq']:.3f} | **{r['d']:+.3f}** |")
        L.append("")
    L.append("**Difference between the two views:** identical PPLs; View 2 "
             "reclassifies **Yi-6B SL256** (−0.025) as a tie and includes "
             "**DeepSeek SL1024** (CARE-KV fell back to BaseQuant) as a loss. "
             "Clearest win either way: **Mistral-7B SL512 (−0.364)**. CARE-KV is "
             "competitive with TurboQuant, not uniformly superior; its robust "
             "advantage is over same-bit BaseQuant INT3 (never worse).")
    out_md = os.path.join(args.odir, "turboquant_comparison_both.md")
    open(out_md, "w").write("\n".join(L) + "\n")
    print("wrote", out_md)

    # figures
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        col = {"win": "#2ca02c", "tie": "#7f7f7f", "loss": "#d62728", "n/a": "#cccccc"}

        def fig(key, title, fname, band):
            rr = [r for r in R if r["d"] is not None]
            labels = [f"{r['m'][:10]}\n{r['sl']}" for r in rr]
            ys = [r["d"] for r in rr]
            cs = [col[r[key]] for r in rr]
            f, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
            bars = ax.bar(range(len(rr)), ys, color=cs, edgecolor="black", linewidth=0.4)
            ax.axhline(0, color="black", lw=0.8)
            if band:
                ax.axhspan(-TOL, TOL, color="#999999", alpha=0.15)
            for i, r in enumerate(rr):
                ax.text(i, ys[i], f"{ys[i]:+.2f}", ha="center",
                        va="bottom" if ys[i] >= 0 else "top", fontsize=7)
            ax.set_xticks(range(len(rr))); ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("ΔPPL  (CARE-KV − TurboQuant)\n↓ below 0 = CARE-KV wins")
            ax.set_title(title)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color=col["win"], label="CARE-KV win"),
                               Patch(color=col["tie"], label="tie"),
                               Patch(color=col["loss"], label="TurboQuant win"),
                               Patch(color=col["n/a"], label="n/a (CARE→base)")],
                      fontsize=8, ncol=4, loc="upper left")
            ax.grid(axis="y", ls="--", alpha=0.3)
            out = os.path.join(args.odir, fname)
            f.savefig(out, dpi=130); plt.close(f); print("wrote", out)

        fig("v1", f"CARE-KV vs TurboQuant (fair INT3) — View 1: margin ±{TOL} "
            f"→ {w1}W/{t1}T/{l1}L", "fig_turboquant_margin002.png", band=True)
        fig("v2", f"CARE-KV vs TurboQuant (fair INT3) — View 2: audited "
            f"→ {w2}W/{t2}T/{l2}L", "fig_turboquant_audited.png", band=False)
    except Exception as e:
        print("figure skipped:", e)

    print(f"View1 {w1}W/{t1}T/{l1}L ({na1} n/a) | View2 {w2}W/{t2}T/{l2}L")


if __name__ == "__main__":
    main()
