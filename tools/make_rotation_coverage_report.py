"""tools/make_rotation_coverage_report.py

Render the rotation coverage ablation (rotation_coverage_ablation_wt2.csv) into
md + figure, with a granularity-aware residual-memory figure (candidate-cap
formula) and the coverage decomposition (finer-granularity vs read-breadth).
DIAGNOSTIC: TinyLlama N=4 SL=128 (ΔPPL<0.5 ~ noise).
"""
from __future__ import annotations
import argparse, csv, math, os

PAPER = "results/paper_eval_20260529_015053"
HD, PAGE, L, HKV = 64, 16, 22, 4
NOISE = 0.5

CFG = {
    "uni_g32v4": ("uniform", 32, 4),
    "uni_g16v2": ("uniform", 16, 2),
    "rot_g32v4": ("Hadamard pre-RoPE", 32, 4),
    "rot_g16v2": ("Hadamard pre-RoPE", 16, 2),
}
ORDER = ["fp16", "base_int3", "uni_g32v4", "uni_g16v2", "rot_g32v4", "rot_g16v2"]
PRIOR_RV2 = {"uni_g32v4": 13.462, "rot_g32v4": 13.259}


def _resmb(kcg, vtb, seqlen=128):
    kcap, vcap = HD // kcg, math.ceil(PAGE / vtb)
    npages = math.ceil(seqlen / PAGE)
    kslot, vslot = PAGE * kcg // 2, vtb * HD // 2
    b = L * HKV * npages * (kcap * kslot + vcap * vslot) \
        + 2 * L * HKV * npages * (kcap + vcap)
    return b / 1024 / 1024, kcap, vcap


def _f(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=PAPER)
    args = ap.parse_args()
    rows = {r["cell"]: r for r in csv.DictReader(
        open(os.path.join(args.paper_dir, "ablations",
                          "rotation_coverage_ablation_wt2.csv")))}
    P = {c: _f(rows[c], "ppl") for c in ORDER if c in rows}

    L_ = ["# Rotation coverage ablation — results (diagnostic)", "",
          "**Question.** After rotation the INT3 residual is diffuse. Does broader "
          "coverage (finer granularity, full-cap store+read) help the rotated base "
          "more than the uniform base? ΔPPL<0.5 ~ noise (N=4, 508 tokens).", "",
          "| cell | base | granularity | K_cap/V_cap | PPL | residual MB |",
          "|---|---|---|---|---:|---:|"]
    for c in ORDER:
        if c not in rows:
            continue
        if c in CFG:
            base, kcg, vtb = CFG[c]
            mb, kc, vc = _resmb(kcg, vtb)
            L_.append(f"| {c} | {base} | g{kcg}v{vtb} | {kc}/{vc} | {P[c]:.4f} | {mb:.3f} |")
        else:
            L_.append(f"| {c} | — | — | — | {P[c]:.4f} | — |")
    L_.append("")
    if all(k in P for k in ("uni_g16v2", "uni_g32v4", "rot_g16v2", "rot_g32v4")):
        du = P["uni_g16v2"] - P["uni_g32v4"]
        dr = P["rot_g16v2"] - P["rot_g32v4"]
        L_.append(f"**Finer granularity (g32v4→g16v2):** uniform {du:+.3f}, rotated "
                  f"{dr:+.3f} (diff-of-diff {dr-du:+.3f}). At full cap residual "
                  f"memory is identical (~{_resmb(32,4)[0]:.2f} MB) → REJECTED.")
        L_.append("")
    if "uni_g32v4" in PRIOR_RV2 and P.get("uni_g32v4") and P.get("rot_g32v4"):
        L_.append(f"**Read-breadth (RV2→4 at g32v4):** uniform "
                  f"{PRIOR_RV2['uni_g32v4']:.3f}→{P['uni_g32v4']:.3f}, rotated "
                  f"{PRIOR_RV2['rot_g32v4']:.3f}→{P['rot_g32v4']:.3f}. Best cell "
                  f"rot_g32v4 (read-all) = {P['rot_g32v4']:.3f}. Lever = read-"
                  "breadth, not subdivision (N=4 noisy → confirm at N=16).")
    out_md = os.path.join(args.paper_dir, "summaries",
                          "rotation_coverage_ablation_results.md")
    open(out_md, "w").write("\n".join(L_) + "\n")
    print("wrote", out_md)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, numpy as np
        grans = ["g32v4", "g16v2"]
        uni = [P["uni_g32v4"], P["uni_g16v2"]]
        rot = [P["rot_g32v4"], P["rot_g16v2"]]
        x = np.arange(2); w = 0.35
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        ax.bar(x - w/2, uni, w, label="uniform base", color="#1f77b4", edgecolor="black")
        ax.bar(x + w/2, rot, w, label="Hadamard pre-RoPE base", color="#2ca02c", edgecolor="black")
        ax.axhline(13.462, color="#d62728", ls="--", lw=1, label="uniform+CARE-KV bar (RV2)=13.462")
        ax.axhline(P["fp16"], color="#7f7f7f", ls=":", lw=1, label=f"fp16={P['fp16']:.2f}")
        ax.set_xticks(x); ax.set_xticklabels(["g32v4 (cap2/4)", "g16v2 (cap4/8)"])
        ax.set_ylabel("WikiText-2 PPL"); ax.legend(fontsize=8)
        ax.set_title("Rotation coverage ablation (TinyLlama N=4 SL=128, read-all) — DIAGNOSTIC")
        ax.grid(axis="y", ls="--", alpha=0.4); ax.set_ylim(12, max(uni+rot)+0.6)
        out_fig = os.path.join(args.paper_dir, "figures", "fig_rotation_coverage_ablation.png")
        fig.savefig(out_fig, dpi=130); plt.close(fig)
        print("wrote", out_fig)
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
