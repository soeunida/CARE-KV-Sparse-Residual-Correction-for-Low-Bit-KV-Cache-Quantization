"""tools/make_turboquant_bit_sweep_figure.py — Part A figures.

Reads ablations/turboquant_bit_sweep_wt2_n4.csv and writes:
  figures/fig_turboquant_bit_sweep_ppl.png
  figures/fig_turboquant_bit_sweep_memory_quality.png
"""
from __future__ import annotations
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    d = {}
    for r in rows:
        try:
            d[r["method_name"]] = dict(
                ppl=float(r["ppl"]) if r["ppl"] not in ("", "0.0", "0") else None,
                mem=float(r["estimated_kv_memory_MB"]) if r["estimated_kv_memory_MB"] else 0.0,
                fam=r["method_family"])
        except (ValueError, KeyError):
            pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default="results/paper_eval_20260529_015053")
    args = ap.parse_args()
    csv_path = os.path.join(args.paper_dir, "ablations", "turboquant_bit_sweep_wt2_n4.csv")
    figs = os.path.join(args.paper_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    d = load(csv_path)
    fp16 = d.get("fp16", {}).get("ppl")

    # ── Figure 1: PPL bars by bit-width ──
    bits = [4, 3, 2]
    series = {
        "BaseQuant": [d.get(f"base_quant_INT{b}", {}).get("ppl") for b in bits],
        "TurboQuant +QJL": [d.get(f"TurboQuant_style_INT{b}", {}).get("ppl") for b in bits],
        "TurboQuant noQJL": [d.get(f"TurboQuant_style_INT{b}_noQJL", {}).get("ppl") for b in bits],
    }
    fig, ax = plt.subplots(figsize=(9, 5.2))
    import numpy as np
    x = np.arange(len(bits)); w = 0.26
    colors = {"BaseQuant": "#bbbbbb", "TurboQuant +QJL": "#d62728", "TurboQuant noQJL": "#f0a0a0"}
    for i, (name, vals) in enumerate(series.items()):
        vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=colors[name], edgecolor="black", lw=0.5)
        for b, v in zip(bars, vals):
            if v: ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    if fp16: ax.axhline(fp16, ls="--", c="#333", lw=1, label=f"fp16 = {fp16:.2f}")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"INT{b}" for b in bits])
    ax.set_ylabel("WT-2 PPL (log scale, lower=better)")
    ax.set_title("TurboQuant-style bit sweep — QJL residual correction\n"
                 "TinyLlama, WT-2 N=4 SL=128 (diagnostic; reimpl, NOT official)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    o1 = os.path.join(figs, "fig_turboquant_bit_sweep_ppl.png")
    plt.savefig(o1, dpi=150); plt.close()
    print("wrote", o1)

    # ── Figure 2: memory vs quality scatter ──
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    pts = [
        ("fp16", "#333333", "o"), ("base_quant_INT4", "#888", "s"),
        ("base_quant_INT3", "#aaa", "s"), ("base_quant_INT2", "#ccc", "s"),
        ("TurboQuant_style_INT4", "#d62728", "^"), ("TurboQuant_style_INT3", "#e8553a", "^"),
        ("TurboQuant_style_INT2", "#f0a0a0", "^"),
        ("CAREKV_fixed_SK2SV4_RK2RV2", "#f2cc8f", "D"),
        ("KIVI_INT3K_INT3V_plus_CAREKV", "#81b29a", "D"),
        ("KVQuantPreRoPE_INT3K_INT3V_plus_CAREKV", "#3d5a80", "D"),
    ]
    for name, c, mk in pts:
        r = d.get(name)
        if not r or r["ppl"] is None or r["ppl"] <= 0: continue
        if r["ppl"] > 100: continue   # skip broken INT2 for readability
        ax.scatter(r["mem"], r["ppl"], c=c, marker=mk, s=90, edgecolor="black", lw=0.6,
                   label=name.replace("_", " ")[:30], zorder=3)
    if fp16: ax.axhline(fp16, ls="--", c="#333", lw=1, alpha=0.6)
    ax.set_xlabel("Estimated KV memory (MB, lower=better)")
    ax.set_ylabel("WT-2 PPL (lower=better)")
    ax.set_title("Memory vs quality — TurboQuant-style vs BaseQuant vs CARE-KV\n"
                 "TinyLlama WT-2 N=4 SL=128 (INT2 cells off-scale, omitted)", fontsize=11)
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    o2 = os.path.join(figs, "fig_turboquant_bit_sweep_memory_quality.png")
    plt.savefig(o2, dpi=150); plt.close()
    print("wrote", o2)


if __name__ == "__main__":
    main()
