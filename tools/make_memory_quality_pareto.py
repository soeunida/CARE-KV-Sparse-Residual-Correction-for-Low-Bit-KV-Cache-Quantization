"""tools/make_memory_quality_pareto.py

Memory-quality Pareto at a long context length: effective KV-cache footprint
(as a fraction of fp16) vs PPL, for fp16 / INT3 base / TurboQuant / CARE-KV.

Answers "does CARE-KV dominate?" HONESTLY: CARE-KV spends more KV memory than a
plain INT3 method (base residual slots) but buys lower PPL — it is a distinct,
higher-quality/higher-memory Pareto point, not a strict domination of TurboQuant.

KV footprint ratios vs fp16 (paper-best config, from tools/analyze_overhead_flops_bw.py):
  fp16      = 1.0
  base_INT3 = (3/8)/2 + (1/32)/2  (INT3 codes + INT8 group scales)   ~0.203
  Turbo_INT3= (3/8)/2             (INT3 codes; QJL adds only a small fixed matrix) ~0.188
  CARE-KV   = base + 4-bit residual slots (SK=2 K + SV=4 V per page)

Reads PPL from results/longctx_ppl/scaling/*.csv.

Usage:
  python tools/make_memory_quality_pareto.py --dataset pg19 --sl 2048 \
    --out results/longctx_ppl/fig_memory_quality_pareto.png
"""
from __future__ import annotations
import argparse, csv, glob, math, os

# paper-best config
P, G, CKG, VTB, B, RB = 16, 32, 32, 4, 3, 4
SK, SV = 2, 4


def kv_ratio(mode, D=128):
    """KV footprint as a fraction of fp16, per token, paper-best config."""
    fp16 = 2 * D * 2                                   # K+V, 2 bytes each
    codes = 2 * D * (B / 8)                            # INT3 packed K+V
    scales = 2 * (D / G) * 1                           # INT8 group scales
    base = codes + scales
    if mode == "fp16":
        return 1.0
    if mode == "turboquant_int3":
        return codes / fp16                            # ~0.188 (QJL matrix ~ negligible/token)
    if mode == "base_quant_int3":
        return base / fp16                             # ~0.203
    if mode in ("carekv_qaware", "carekv_stored_int3", "carekv_magnitude"):
        # residual slots amortized per token: per page (P tokens) we store
        # SK K-slots (P*CKG 4-bit) + SV V-slots (VTB*D 4-bit), each + fp16 scale.
        k_slot = P * CKG * RB / 8 + 2
        v_slot = VTB * D * RB / 8 + 2
        resid_per_tok = (SK * k_slot + SV * v_slot) / P
        return (base + resid_per_tok) / fp16
    raise ValueError(mode)


def load_ppl(csvs, dataset, sl):
    out = {}
    for p in csvs:
        for r in csv.DictReader(open(p)):
            if r.get("status") == "real" and r["dataset"] == dataset and int(r["seq_len"]) == sl:
                try:
                    out[r["mode"]] = float(r["ppl"])
                except (ValueError, KeyError):
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+",
                    default=glob.glob("results/longctx_ppl/scaling/*.csv"))
    ap.add_argument("--dataset", default="pg19")
    ap.add_argument("--sl", type=int, default=2048)
    ap.add_argument("--out", default="results/longctx_ppl/fig_memory_quality_pareto.png")
    A = ap.parse_args()

    ppl = load_ppl(A.csv, A.dataset, A.sl)
    order = ["fp16", "base_quant_int3", "turboquant_int3", "carekv_qaware"]
    pts = []
    for m in order:
        if m in ppl:
            pts.append((m, kv_ratio(m), ppl[m]))
    if not pts:
        print(f"[pareto] no PPL for {A.dataset} SL{A.sl}"); return
    print(f"[pareto] {A.dataset} SL{A.sl}:")
    for m, x, y in pts:
        print(f"  {m:20s} KVmem={x:.3f}x fp16   PPL={y:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[pareto] matplotlib unavailable: {e}"); return
    labels = {"fp16": "fp16", "base_quant_int3": "INT3 base",
              "turboquant_int3": "TurboQuant", "carekv_qaware": "CARE-KV"}
    colors = {"fp16": "k", "base_quant_int3": "C0",
              "turboquant_int3": "C1", "carekv_qaware": "C2"}
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for m, x, y in pts:
        ax.scatter([x], [y], s=90, color=colors.get(m, "C3"), zorder=3)
        ax.annotate(labels.get(m, m), (x, y), textcoords="offset points",
                    xytext=(8, 6), fontsize=10)
    ax.set_xlabel("effective KV-cache memory  (fraction of fp16)  →  smaller is better")
    ax.set_ylabel("PPL  →  lower is better")
    ax.set_title(f"Memory–quality Pareto ({A.dataset.upper()}, SL={A.sl}, Mistral-7B)\n"
                 "CARE-KV = higher-memory, higher-quality point (not a strict Turbo domination)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(A.out, dpi=130, bbox_inches="tight")
    print(f"[pareto] wrote {A.out}")


if __name__ == "__main__":
    main()
