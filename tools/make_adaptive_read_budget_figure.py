"""tools/make_adaptive_read_budget_figure.py

Reads tools/eval_adaptive_read_budget.py's output CSV and produces a
two-panel comparison figure:

  Top panel:    PPL bar chart, fixed vs adaptive cells side by side
  Bottom panel: effective_RK_mean + effective_RV_mean per cell
                (shows how much the threshold actually shrinks the read budget)

Writes: --out PNG
"""
from __future__ import annotations
import argparse, csv, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(p):
    if not os.path.exists(p): return []
    with open(p, newline="") as f: return list(csv.DictReader(f))


def _fnum(r, k, d=0.0):
    try: return float(r.get(k, d))
    except (TypeError, ValueError): return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = _read(args.csv)
    rows = [r for r in rows if "ppl" in r and r.get("ppl")]
    if not rows:
        print(f"[fig] no rows in {args.csv}")
        return

    labels = [r["label"] for r in rows]
    ppls   = [_fnum(r, "ppl") for r in rows]
    eff_rk = [_fnum(r, "effective_RK_mean") for r in rows]
    eff_rv = [_fnum(r, "effective_RV_mean") for r in rows]
    modes  = [r.get("mode", "absolute") for r in rows]

    colors = []
    for lbl, mode in zip(labels, modes):
        if "RK2_RV2" in lbl and mode == "absolute":
            colors.append("#2ca02c")    # paper-best (green)
        elif mode == "adaptive_score":
            colors.append("#9467bd")    # adaptive (purple)
        else:
            colors.append("#888888")    # other fixed (grey)

    # Paper-best reference (RK=RV=2 fixed)
    bestline = next((p for l, p in zip(labels, ppls)
                     if "fixed_RK2_RV2" in l), None)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True,
                              gridspec_kw={"height_ratios": [1.3, 1.0]})

    # Top panel: PPL
    bars = axes[0].bar(range(len(labels)), ppls,
                        color=colors, edgecolor="black", linewidth=0.4)
    for b, p in zip(bars, ppls):
        axes[0].text(b.get_x() + b.get_width() / 2, p, f"{p:.3f}",
                      ha="center", va="bottom", fontsize=8)
    if bestline is not None:
        axes[0].axhline(y=bestline, color="#2ca02c", linestyle="--",
                         linewidth=0.7, alpha=0.6,
                         label=f"paper-best fixed RK=RV=2 = {bestline:.3f}")
        axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("PPL  (lower = better)", fontsize=11)
    # Title infers dataset/SL from the CSV's first non-base row
    ds = next((r.get("dataset", "synthetic") for r in rows
               if r.get("dataset")), "synthetic")
    sl = next((r.get("seq_len", "?") for r in rows if r.get("seq_len")), "?")
    n_samples = next((r.get("num_samples", "") for r in rows
                       if r.get("num_samples")), "")
    n_blurb = f" N={n_samples}" if n_samples and ds == "wikitext" else ""
    ds_pretty = "WikiText-2" if ds == "wikitext" else "synthetic"
    axes[0].set_title(f"Adaptive vs fixed read budget — PPL "
                      f"({ds_pretty}{n_blurb} SL={sl}, INT3, SK=2 SV=4)",
                      fontsize=13)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Bottom panel: effective read budget
    import numpy as np
    x = np.arange(len(labels))
    w = 0.4
    axes[1].bar(x - w/2, eff_rk, width=w, color="#1f77b4",
                 edgecolor="black", linewidth=0.3, label="effective_RK_mean")
    axes[1].bar(x + w/2, eff_rv, width=w, color="#ff7f0e",
                 edgecolor="black", linewidth=0.3, label="effective_RV_mean")
    for i, (k, v) in enumerate(zip(eff_rk, eff_rv)):
        axes[1].text(i - w/2, k, f"{k:.1f}", ha="center", va="bottom",
                      fontsize=7)
        axes[1].text(i + w/2, v, f"{v:.1f}", ha="center", va="bottom",
                      fontsize=7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[1].set_ylabel("mean effective reads per query", fontsize=11)
    axes[1].set_title("Effective per-cell read budget (after threshold filter)",
                       fontsize=12)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9, loc="upper left")

    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
