"""Single combined figure for the KVQuant-style + CARE-KV unblock study:
   fig_kvquant_carekv_unblock.png
     left  : PPL bar chart (fp16 / base_quant_INT3 reference lines)
     right : memory-quality scatter (log-y)
"""
from __future__ import annotations
import argparse, csv, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FAMILY_COLORS = {
    "fp16":                  "#1f77b4",
    "base_quant":            "#888888",
    "care_kv":               "#2ca02c",
    "kivi_style":            "#ff7f0e",
    "kivi_plus_carekv":      "#9467bd",
    "kvquant_style":         "#bcbd22",
    "kvquant_plus_carekv":   "#d62728",
    "rotatekv_style":        "#17becf",
    "rotatekv_plus_carekv":  "#e377c2",
}


def _read(p):
    if not os.path.exists(p): return []
    with open(p, newline="") as f: return list(csv.DictReader(f))


def _f(r, k, d=0.0):
    try: return float(r.get(k, d))
    except (TypeError, ValueError): return d


def _valid(rows):
    return [r for r in rows if r.get("ppl") and _f(r, "ppl") > 0]


def make_figure(rows, out, label):
    rows = _valid(rows)
    if not rows:
        print(f"[fig] no valid rows -> skip {out}")
        return
    labels = [r["method_name"] for r in rows]
    ppls   = [_f(r, "ppl") for r in rows]
    colors = [FAMILY_COLORS.get(r.get("method_family", ""), "#444") for r in rows]
    fp16 = next((p for r, p in zip(rows, ppls) if r["method_family"] == "fp16"), None)
    int3 = next((p for r, p in zip(rows, ppls) if r["method_name"] == "base_quant_INT3"), None)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # ── left: PPL bars ──
    bars = axL.bar(range(len(labels)), ppls, color=colors,
                   edgecolor="black", linewidth=0.5)
    for b, p in zip(bars, ppls):
        axL.text(b.get_x() + b.get_width()/2, p, f"{p:.3f}",
                 ha="center", va="bottom", fontsize=8)
    if fp16: axL.axhline(y=fp16, color="#1f77b4", linestyle="--", linewidth=0.8,
                         alpha=0.6, label=f"fp16 = {fp16:.3f}")
    if int3: axL.axhline(y=int3, color="#888888", linestyle=":", linewidth=0.8,
                         alpha=0.6, label=f"base_quant_INT3 = {int3:.3f}")
    axL.legend(fontsize=9, loc="upper left")
    axL.set_xticks(range(len(labels)))
    axL.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axL.set_ylabel("PPL  (lower = better)", fontsize=11)
    axL.set_title("PPL by method", fontsize=12)
    axL.grid(axis="y", linestyle="--", alpha=0.4)

    # ── right: memory-quality scatter ──
    for r in rows:
        ppl = _f(r, "ppl"); mem = _f(r, "estimated_kv_memory_MB")
        if ppl <= 0 or mem <= 0: continue
        fam = r.get("method_family", "")
        color = FAMILY_COLORS.get(fam, "#444")
        marker = "*" if fam == "fp16" else ("o" if "carekv" in fam else "s")
        size = 220 if fam == "fp16" else (150 if "carekv" in fam else 90)
        axR.scatter(mem, ppl, c=color, s=size, marker=marker, alpha=0.85,
                    edgecolors="black", linewidths=0.5)
        axR.annotate(r["method_name"], (mem, ppl), fontsize=7.5,
                     xytext=(5, 4), textcoords="offset points")
    axR.set_xlabel("Estimated KV-cache memory (MB)  (lower = better)", fontsize=11)
    axR.set_ylabel("PPL  (log, lower = better)", fontsize=11)
    axR.set_yscale("log")
    axR.set_title("Memory–quality trade-off", fontsize=12)
    axR.grid(linestyle="--", alpha=0.4, which="both")

    fig.suptitle(f"KVQuant-style (pre-RoPE) + CARE-KV unblock — {label}",
                 fontsize=13)
    fig.text(0.5, 0.002,
             "KVQuant-style is a same-condition reimplementation (NOT official KVQuant). "
             "pre-RoPE K quantization stacked with CARE-KV residual correction.",
             ha="center", fontsize=8, style="italic", color="#666")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = _read(args.csv)
    if not rows:
        print(f"[fig] no rows in {args.csv}"); return
    ds = rows[0].get("dataset", "synthetic")
    sl = rows[0].get("seq_len", "?")
    n = rows[0].get("num_samples", "")
    ds_p = "WikiText-2" if ds == "wikitext" else "synthetic"
    nb = f" N={n}" if n and ds == "wikitext" else ""
    label = f"{ds_p}{nb} SL={sl}, TinyLlama-1.1B"
    make_figure(rows, args.out, label)


if __name__ == "__main__":
    main()
