"""tools/make_prefilter_figure.py — figure for the router pre-filter C sweep
(post-fix). Plots ΔPPL vs the exact-scored fraction (BW proxy) for the magnitude
bound and the sign-sketch proxy at two seq_lens, showing the magnitude bound
holds PPL while restricting the O(S) sketch read, and the sign proxy is not
better. Data: prefilter_fix_{mag,sign32}_sl{512,1024}.csv.
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "results/router_diagnostic"


def load(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def series(path):
    rows = [r for r in load(path) if r.get("scored_frac")]
    frac = [float(r["scored_frac"]) * 100 for r in rows]
    dppl = [float(r["dppl_vs_exact"] or 0) for r in rows]
    return frac, dppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{D}/fig_prefilter_sweep.png")
    args = ap.parse_args()

    curves = [
        (f"{D}/prefilter_fix_mag_sl512.csv",   "magnitude, SL512",  "#1f77b4", "o-"),
        (f"{D}/prefilter_fix_mag_sl1024.csv",  "magnitude, SL1024", "#1f77b4", "s--"),
        (f"{D}/prefilter_fix_sign32_sl512.csv", "sign b=32, SL512",  "#d62728", "o-"),
        (f"{D}/prefilter_fix_sign32_sl1024.csv","sign b=32, SL1024", "#d62728", "s--"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for path, lab, color, style in curves:
        if not os.path.exists(path):
            continue
        frac, dppl = series(path)
        ax.plot(frac, dppl, style, color=color, label=lab, alpha=0.9)

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.axhspan(-0.05, 0.05, color="green", alpha=0.08, label="±0.05 PPL (noise)")
    ax.set_xlabel("K candidates exact-scored (% of pool)   ← more BW saved")
    ax.set_ylabel("Δ PPL vs exact router")
    ax.invert_xaxis()
    ax.set_title("Router pre-filter (post-fix): magnitude bound holds PPL at "
                 "12.5% kept;\nsign proxy is noisier, not better")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
