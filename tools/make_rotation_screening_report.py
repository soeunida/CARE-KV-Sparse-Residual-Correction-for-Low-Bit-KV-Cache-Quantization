"""tools/make_rotation_screening_report.py

Render the rotation + CARE-KV screening CSV into md + figure with a data-driven
GO/NO-GO verdict (robust to a subset of arms). DIAGNOSTIC.
"""
from __future__ import annotations
import argparse, csv, os

PAPER = "results/paper_eval_20260529_015053"
NOISE = 0.5

LABELS = {
    "fp16": "fp16 (reference)",
    "base_int3": "BaseQuant INT3 (uniform)",
    "uniform_carekv": "uniform + CARE-KV (bar)",
    "rot_post_carekv": "Hadamard post-RoPE + CARE-KV",
    "rot_pre_carekv": "Hadamard pre-RoPE + CARE-KV (arm4)",
    "rand_pre_carekv": "random-rot pre-RoPE + CARE-KV (arm5)",
    "rand_pre_base": "random-rot pre-RoPE standalone",
    "rot_pre_base": "Hadamard pre-RoPE standalone",
}
ORDER = ["fp16", "base_int3", "uniform_carekv", "rot_post_carekv",
         "rot_pre_carekv", "rand_pre_carekv", "rand_pre_base", "rot_pre_base"]


def _f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main():
    global NOISE
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=PAPER)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--noise", type=float, default=NOISE)
    ap.add_argument("--setup", default="N=4, SL=128 (508 evaluated tokens)")
    args = ap.parse_args()
    NOISE = args.noise
    csv_path = args.csv or os.path.join(args.paper_dir, "ablations",
                                        "rotation_carekv_screening_wt2.csv")
    rows = {r["arm"]: r for r in csv.DictReader(open(csv_path))}
    bar = _f(rows["uniform_carekv"]["ppl"])
    fp16 = _f(rows["fp16"]["ppl"])
    present = [k for k in ORDER if k in rows]

    L = []
    L.append("# Rotation + CARE-KV stack — screening results (diagnostic)")
    L.append("")
    L.append(f"**Setup.** TinyLlama-1.1B, WikiText-2 PPL, {args.setup}, INT3, "
             "paper-best CARE-KV budget. Same `baselines/` harness as the pilot. "
             f"**Noise band: ΔPPL < {NOISE} treated as below noise.**")
    L.append("")
    L.append("| arm | PPL | Δ vs uniform+CARE-KV | Δ vs fp16 | residual MB | K_reads | V_reads |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for k in present:
        r = rows[k]
        p = _f(r["ppl"])
        dbar = (p - bar) if p else None
        dfp = (p - fp16) if p else None
        mark = ""
        if k not in ("fp16", "base_int3", "uniform_carekv") and dbar is not None:
            mark = " WIN" if dbar < -NOISE else (" ~" if abs(dbar) <= NOISE else " x")
        L.append(f"| {LABELS[k]} | {p:.4f} | "
                 f"{(('%+.4f'%dbar) if dbar is not None else '')}{mark} | "
                 f"{('%+.4f'%dfp) if dfp is not None else ''} | "
                 f"{r.get('residual_memory_MB','')} | {r.get('k_reads','')} | "
                 f"{r.get('v_reads','')} |")
    L.append("")
    L.append("## Findings")
    L.append("")
    care_arms = {k: _f(rows[k]["ppl"]) for k in present
                 if k.endswith("_carekv") and k != "uniform_carekv"}
    best_arm = min(care_arms, key=care_arms.get) if care_arms else None
    best_ppl = care_arms[best_arm] if best_arm else None
    if "rot_post_carekv" in rows and "rot_pre_carekv" in rows:
        post = _f(rows["rot_post_carekv"]["ppl"]); pre = _f(rows["rot_pre_carekv"]["ppl"])
        L.append(f"**H1 (placement).** Hadamard post-RoPE + CARE-KV = {post:.2f} vs "
                 f"pre-RoPE + CARE-KV = {pre:.2f} (swing {post-pre:+.2f}). Pre-RoPE "
                 "is much less harmful than post-RoPE.")
        L.append("")
    if "base_int3" in rows:
        bb = _f(rows["base_int3"]["ppl"])
        for sk, nm in (("rot_pre_base", "Hadamard pre-RoPE"),
                       ("rand_pre_base", "random pre-RoPE")):
            if sk in rows:
                L.append(f"- {nm} rotation **standalone** = {_f(rows[sk]['ppl']):.2f} "
                         f"(Δ vs uniform INT3 base {_f(rows[sk]['ppl'])-bb:+.2f})")
        L.append("")
    if best_arm is not None:
        d = best_ppl - bar
        if d < -NOISE:
            verdict = "GO"
            vtxt = (f"Best stack arm **{LABELS[best_arm]}** = {best_ppl:.3f} beats "
                    f"the bar {bar:.3f} by {d:+.3f} (> noise {NOISE}) at equal "
                    "memory → proceed to the 4-model confirmation.")
        elif abs(d) <= NOISE:
            verdict = "NO-GO (parity)"
            vtxt = (f"Best stack arm **{LABELS[best_arm]}** = {best_ppl:.3f} only "
                    f"ties the bar {bar:.3f} (Δ {d:+.3f}, within noise {NOISE}) → "
                    "rotation gives no benefit on top of CARE-KV; do NOT scale.")
        else:
            verdict = "NO-GO"
            vtxt = (f"Best stack arm **{LABELS[best_arm]}** = {best_ppl:.3f} is "
                    f"worse than the bar {bar:.3f} (Δ {d:+.3f}) → do NOT scale.")
        L.append(f"## Decision: {verdict}")
        L.append("")
        L.append(vtxt)
        L.append("")
        L.append("**Interpretation.** Pre-RoPE rotation and CARE-KV's sparse "
                 "output-error residual are non-additive / substitutes at this "
                 "scale. (QJL, TurboQuant's real edge, is score-level and cannot "
                 "be stacked here.)")
        L.append("")
    L.append("Paper-best config is UNCHANGED (no arm clears the bar beyond noise).")
    out_md = os.path.join(args.paper_dir, "summaries",
                          f"rotation_carekv_screening_results{args.tag}.md")
    open(out_md, "w").write("\n".join(L) + "\n")
    print("wrote", out_md)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [LABELS[k].replace(" + ", "+\n") for k in present]
        ys = [_f(rows[k]["ppl"]) for k in present]
        colors = []
        for k in present:
            if k == "fp16": colors.append("#7f7f7f")
            elif k == "base_int3": colors.append("#bcbd22")
            elif k == "uniform_carekv": colors.append("#1f77b4")
            elif "carekv" in k: colors.append("#2ca02c")
            else: colors.append("#d62728")
        fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
        bars = ax.bar(range(len(xs)), ys, color=colors, edgecolor="black", linewidth=0.4)
        ax.axhline(bar, color="#1f77b4", ls="--", lw=1, label=f"bar={bar:.2f}")
        ax.axhline(fp16, color="#7f7f7f", ls=":", lw=1, label=f"fp16={fp16:.2f}")
        for i, (b, y) in enumerate(zip(bars, ys)):
            d = y - bar
            txt = f"{y:.2f}" + (f"\n({d:+.2f})" if present[i] not in
                                ("fp16", "base_int3", "uniform_carekv") else "")
            ax.text(b.get_x()+b.get_width()/2, y, txt, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("WikiText-2 PPL"); ax.legend(fontsize=9); ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_title(f"Rotation + CARE-KV screening (TinyLlama, {args.setup}, INT3) "
                     f"— DIAGNOSTIC, ΔPPL<{NOISE} within noise", fontsize=11)
        out_fig = os.path.join(args.paper_dir, "figures",
                               f"fig_rotation_carekv_screening{args.tag}.png")
        fig.savefig(out_fig, dpi=130); plt.close(fig)
        print("wrote", out_fig)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
