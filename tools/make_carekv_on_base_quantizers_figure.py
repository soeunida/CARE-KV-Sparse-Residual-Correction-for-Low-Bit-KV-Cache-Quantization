"""Two figures for Phase Q (CARE-KV on top of base quantizers)."""
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


FAMILY_COLORS = {
    "fp16":             "#1f77b4",
    "base_quant":       "#888888",
    "care_kv":          "#2ca02c",
    "kivi_style":       "#ff7f0e",
    "kivi_plus_carekv": "#9467bd",
}


def _supported(rows):
    return [r for r in rows
            if r.get("official_or_reimpl") != "unsupported"
            and r.get("ppl") and float(r["ppl"]) > 0]


def fig_ppl(rows, out, label):
    rows = _supported(rows)
    if not rows: return
    labels = [r["method_name"] for r in rows]
    ppls   = [_fnum(r, "ppl") for r in rows]
    colors = [FAMILY_COLORS.get(r.get("method_family", ""), "#444") for r in rows]
    fp16 = next((p for r, p in zip(rows, ppls) if r["method_family"] == "fp16"), None)
    int3 = next((p for r, p in zip(rows, ppls) if r["method_name"] == "base_quant_INT3"), None)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    bars = ax.bar(range(len(labels)), ppls, color=colors,
                   edgecolor="black", linewidth=0.4)
    for b, p in zip(bars, ppls):
        ax.text(b.get_x() + b.get_width()/2, p, f"{p:.3f}",
                ha="center", va="bottom", fontsize=8)
    if fp16: ax.axhline(y=fp16, color="#1f77b4", linestyle="--", linewidth=0.7,
                         alpha=0.6, label=f"fp16 = {fp16:.3f}")
    if int3: ax.axhline(y=int3, color="#888888", linestyle=":", linewidth=0.7,
                         alpha=0.6, label=f"base_quant_INT3 = {int3:.3f}")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("PPL  (lower = better)", fontsize=11)
    ax.set_title(f"Phase Q — CARE-KV on top of base quantizers ({label})",
                 fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_memory_quality(rows, out, label):
    rows = _supported(rows)
    if not rows: return
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for r in rows:
        ppl = _fnum(r, "ppl"); mem = _fnum(r, "estimated_kv_memory_MB")
        if ppl <= 0 or mem <= 0: continue
        fam = r.get("method_family", "")
        color = FAMILY_COLORS.get(fam, "#444")
        marker = "*" if fam == "fp16" else ("o" if "care_kv" in fam else "s")
        size = 180 if fam in ("fp16", "care_kv") else 80
        ax.scatter(mem, ppl, c=color, s=size, marker=marker, alpha=0.85,
                   edgecolors="black", linewidths=0.4)
        ax.annotate(r["method_name"], (mem, ppl), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Estimated KV-cache memory (MB)  (lower = better)", fontsize=11)
    ax.set_ylabel("PPL  (log, lower = better)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(f"Memory–quality scatter ({label})", fontsize=12)
    ax.grid(linestyle="--", alpha=0.4, which="both")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ppl-out", required=True)
    ap.add_argument("--mem-out", required=True)
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
    fig_ppl(rows, args.ppl_out, label)
    fig_memory_quality(rows, args.mem_out, label)


if __name__ == "__main__":
    main()
