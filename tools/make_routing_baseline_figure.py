"""tools/make_routing_baseline_figure.py

Plot the routing-baseline ablation as a grouped bar chart:
  x-axis: baseline label
  y-axis: PPL (lower = better)
  bar color: green for paper-best (carekv_score), grey for others,
             dashed line for base_quant reference

Reads:  results/.../ablations/routing_baseline_ablation.csv
Writes: results/.../figures/fig_routing_baseline_ablation.png
"""
from __future__ import annotations
import argparse, csv, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = _read_csv(args.csv)
    if not rows:
        print(f"[fig] no rows in {args.csv}")
        return

    order = ["base_quant", "random", "magnitude_only",
             "attention_only", "carekv_score", "oracle_proxy"]
    rows_by_label = {r["label"]: r for r in rows if "ppl" in r}
    xs = [lbl for lbl in order if lbl in rows_by_label]
    ys = [float(rows_by_label[lbl]["ppl"]) for lbl in xs]

    colors = []
    for lbl in xs:
        if lbl == "carekv_score":
            colors.append("#2ca02c")    # green (paper-best)
        elif lbl == "oracle_proxy":
            colors.append("#9467bd")    # purple (diagnostic upper bound)
        elif lbl == "base_quant":
            colors.append("#d62728")    # red (no correction)
        else:
            colors.append("#7f7f7f")    # grey (other baselines)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(xs, ys, color=colors, edgecolor="black", linewidth=0.4)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2.0, y, f"{y:.3f}",
                ha="center", va="bottom", fontsize=9)

    # base_quant reference line
    if "base_quant" in rows_by_label:
        bq = float(rows_by_label["base_quant"]["ppl"])
        ax.axhline(y=bq, color="#d62728", linestyle="--", linewidth=0.7, alpha=0.6,
                   label=f"base_quant = {bq:.3f}")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_ylabel("PPL  (lower = better)", fontsize=11)
    ax.set_title("Routing baseline ablation — same store/read budget across cells",
                 fontsize=13)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {args.out}")


if __name__ == "__main__":
    main()
