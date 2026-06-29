"""tools/make_budget_figures.py

Phase N — figures for the budget-experiments suite.

Reads the 5 CSVs from eval_budget_experiments.py and produces 6 figures:
  fig_budget_ratio_vs_absolute.png   (A1 + A2 side-by-side)
  fig_store_budget_sweep.png         (B)
  fig_read_budget_sweep.png          (C)
  fig_kv_budget_balance.png          (D)
  fig_budget_pareto.png              (PPL vs cache_mem_MB scatter across A2+B+C+D)
  fig_budget_granularity_sweep.png   (granularity sensitivity; DIAGNOSTIC)
"""
from __future__ import annotations
import argparse, csv, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fnum(r, key, default=0.0):
    try:
        return float(r.get(key, default))
    except (TypeError, ValueError):
        return default


def fig_ratio_vs_absolute(rows_a1, rows_a2, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    # A1: ratio grid — heatmap-style PPL by (store_ratio, read_ratio)
    if rows_a1:
        srs = sorted({float(r["store_budget_ratio"]) for r in rows_a1})
        rrs = sorted({float(r["read_budget_ratio"]) for r in rows_a1})
        grid = [[None] * len(rrs) for _ in srs]
        for r in rows_a1:
            si = srs.index(float(r["store_budget_ratio"]))
            ri = rrs.index(float(r["read_budget_ratio"]))
            grid[si][ri] = _fnum(r, "ppl")
        import numpy as np
        A = np.array([[v if v is not None else float("nan") for v in row]
                       for row in grid])
        im = axes[0].imshow(A, cmap="viridis", aspect="auto",
                             interpolation="nearest")
        axes[0].set_xticks(range(len(rrs)))
        axes[0].set_xticklabels([f"{r:.2f}" for r in rrs])
        axes[0].set_yticks(range(len(srs)))
        axes[0].set_yticklabels([f"{s:.2f}" for s in srs])
        axes[0].set_xlabel("read_budget_ratio")
        axes[0].set_ylabel("store_budget_ratio")
        axes[0].set_title("A1: Ratio budget grid (PPL)")
        for i in range(len(srs)):
            for j in range(len(rrs)):
                if A[i, j] == A[i, j]:
                    axes[0].text(j, i, f"{A[i, j]:.2f}", ha="center",
                                  va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=axes[0], shrink=0.8, label="PPL")
    else:
        axes[0].set_title("A1: (no data)")

    # A2: absolute grid — bar chart with labels on x-axis
    if rows_a2:
        xs = [r["label"].replace("abs_", "") for r in rows_a2]
        ys = [_fnum(r, "ppl") for r in rows_a2]
        bars = axes[1].bar(xs, ys, color="#2ca02c", edgecolor="black",
                            linewidth=0.4)
        for b, y in zip(bars, ys):
            axes[1].text(b.get_x() + b.get_width() / 2, y, f"{y:.3f}",
                          ha="center", va="bottom", fontsize=8)
        axes[1].set_xticks(range(len(xs)))
        axes[1].set_xticklabels(xs, rotation=35, ha="right", fontsize=8)
        axes[1].set_ylabel("PPL")
        axes[1].set_title("A2: Absolute budget grid (PPL)")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    else:
        axes[1].set_title("A2: (no data)")

    fig.suptitle("Ratio vs absolute budget (synthetic, SL=64, INT3)",
                 fontsize=14)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_store_sweep(rows, out):
    if not rows:
        return
    xs = [r["label"].replace("store_", "") for r in rows]
    ys = [_fnum(r, "ppl") for r in rows]
    mems = [_fnum(r, "cache_mem_MB") for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = ax.bar(xs, ys, color="#1f77b4", edgecolor="black", linewidth=0.4)
    for b, y, m in zip(bars, ys, mems):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.3f}\n{m:.1f} MB",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=25, ha="right")
    ax.set_ylabel("PPL  (lower = better)")
    ax.set_title("B: Store budget sweep (RK=RV=2 fixed; synthetic SL=64)",
                 fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_read_sweep(rows, out):
    if not rows:
        return
    xs = [r["label"].replace("read_", "") for r in rows]
    ys = [_fnum(r, "ppl") for r in rows]
    kreads = [_fnum(r, "K_reads") for r in rows]
    vreads = [_fnum(r, "V_reads") for r in rows]
    totalreads = [k + v for k, v in zip(kreads, vreads)]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(xs, ys, "o-", color="#1f77b4", markersize=8)
    for i, y in enumerate(ys):
        axes[0].annotate(f"{y:.3f}", (i, y), textcoords="offset points",
                          xytext=(0, 6), ha="center", fontsize=9)
    axes[0].set_ylabel("PPL  (lower = better)")
    axes[0].set_title("C: Read budget sweep (SK=2 SV=4 fixed; synthetic SL=64)",
                       fontsize=13)
    axes[0].grid(linestyle="--", alpha=0.4)

    axes[1].bar(range(len(xs)), totalreads, color="#888888", alpha=0.6,
                 label="K+V reads")
    axes[1].set_xticks(range(len(xs)))
    axes[1].set_xticklabels(xs, rotation=20, ha="right")
    axes[1].set_ylabel("total residual reads")
    axes[1].set_title("Total residual reads per cell")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_kv_balance(rows, out):
    if not rows:
        return
    xs = [r["label"].replace("balance_", "") for r in rows]
    ys = [_fnum(r, "ppl") for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = ax.bar(xs, ys, color="#9467bd", edgecolor="black", linewidth=0.4)
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("PPL  (lower = better)")
    ax.set_title("D: K/V budget balance (synthetic SL=64, INT3)",
                 fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_pareto(rows_all, out):
    if not rows_all:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for r in rows_all:
        ppl = _fnum(r, "ppl")
        mem = _fnum(r, "cache_mem_MB")
        if ppl <= 0 or mem <= 0:
            continue
        label = r.get("label", "")
        is_winner = "winner" in label.lower() or "balanced_2_4_2_2" in label or label == "abs_SK2_SV4_RK2_RV2"
        color = "#2ca02c" if is_winner else "#888888"
        size = 80 if is_winner else 40
        ax.scatter(mem, ppl, c=color, s=size, alpha=0.7, edgecolors="black",
                    linewidths=0.4)
        if is_winner or "RK0_RV0" in label or "SK0" in label or "abs_SK1_SV1_RK1_RV1" in label:
            ax.annotate(label, (mem, ppl), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Estimated KV-cache memory (MB)")
    ax.set_ylabel("PPL  (lower = better)")
    ax.set_title("Budget Pareto: PPL vs cache memory across all experiments "
                 "(synthetic SL=64)", fontsize=12)
    ax.grid(linestyle="--", alpha=0.4)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_granularity_sweep(rows, out):
    """Residual-granularity sensitivity (DIAGNOSTIC).

    Left: PPL heatmap over (k_channel_group rows × v_token_block cols), with the
    per-cell K/V candidate caps annotated.  Right: PPL vs residual memory (MB),
    showing whether finer residuals (more candidates, more memory) buy lower PPL.
    The paper-best cell (kcg=32, vtb=4 → caps 2×4) is highlighted.
    """
    if not rows:
        return
    import numpy as np
    kcgs = sorted({int(r["k_channel_group"]) for r in rows}, reverse=True)
    vtbs = sorted({int(r["v_token_block"]) for r in rows}, reverse=True)
    P = [[float("nan")] * len(vtbs) for _ in kcgs]
    caps = [[("", "")] * len(vtbs) for _ in kcgs]
    for r in rows:
        i = kcgs.index(int(r["k_channel_group"]))
        j = vtbs.index(int(r["v_token_block"]))
        P[i][j] = _fnum(r, "ppl")
        caps[i][j] = (r.get("K_cand_cap", ""), r.get("V_cand_cap", ""))
    A = np.array(P)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    im = axes[0].imshow(A, cmap="viridis_r", aspect="auto", interpolation="nearest")
    axes[0].set_xticks(range(len(vtbs)))
    axes[0].set_xticklabels([str(v) for v in vtbs])
    axes[0].set_yticks(range(len(kcgs)))
    axes[0].set_yticklabels([str(k) for k in kcgs])
    axes[0].set_xlabel("v_token_block  (smaller → more V candidates)")
    axes[0].set_ylabel("k_channel_group  (smaller → more K candidates)")
    axes[0].set_title("Granularity sweep: PPL (lower=better)\nannotated with K_cap×V_cap")
    for i in range(len(kcgs)):
        for j in range(len(vtbs)):
            if A[i, j] == A[i, j]:
                kc, vc = caps[i][j]
                axes[0].text(j, i, f"{A[i, j]:.3f}\n{kc}x{vc}", ha="center",
                             va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="PPL")

    for r in rows:
        ppl = _fnum(r, "ppl")
        mem = _fnum(r, "residual_mem_MB")
        if ppl <= 0:
            continue
        paper = int(r["k_channel_group"]) == 32 and int(r["v_token_block"]) == 4
        axes[1].scatter(mem, ppl, s=110 if paper else 55,
                        c="#d62728" if paper else "#1f77b4",
                        edgecolors="black", linewidths=0.4, zorder=3)
        axes[1].annotate(f"{r.get('K_cand_cap','')}x{r.get('V_cand_cap','')}"
                         + ("  (paper)" if paper else ""),
                         (mem, ppl), fontsize=7, xytext=(4, 3),
                         textcoords="offset points")
    axes[1].set_xlabel("Residual memory (MB, effective store budget)")
    axes[1].set_ylabel("PPL  (lower = better)")
    axes[1].set_title("PPL vs residual memory across granularities")
    axes[1].grid(linestyle="--", alpha=0.4)

    fig.suptitle("Residual granularity sensitivity (DIAGNOSTIC; synthetic SL=128, INT3)",
                 fontsize=13)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations-dir", required=True)
    ap.add_argument("--figures-dir",   required=True)
    args = ap.parse_args()

    A1 = _read(os.path.join(args.ablations_dir, "budget_ratio_grid.csv"))
    A2 = _read(os.path.join(args.ablations_dir, "budget_absolute_grid.csv"))
    B  = _read(os.path.join(args.ablations_dir, "store_budget_sweep.csv"))
    C  = _read(os.path.join(args.ablations_dir, "read_budget_sweep.csv"))
    D  = _read(os.path.join(args.ablations_dir, "kv_budget_balance.csv"))

    os.makedirs(args.figures_dir, exist_ok=True)
    fig_ratio_vs_absolute(A1, A2, os.path.join(args.figures_dir,
                                                "fig_budget_ratio_vs_absolute.png"))
    fig_store_sweep(B, os.path.join(args.figures_dir, "fig_store_budget_sweep.png"))
    fig_read_sweep(C, os.path.join(args.figures_dir, "fig_read_budget_sweep.png"))
    fig_kv_balance(D, os.path.join(args.figures_dir, "fig_kv_budget_balance.png"))

    # Pareto: combine A2 + B + C + D (skip A1 which uses different budget mode)
    all_rows = A2 + B + C + D
    fig_pareto(all_rows, os.path.join(args.figures_dir, "fig_budget_pareto.png"))

    # Granularity sensitivity sweep (diagnostic)
    G = _read(os.path.join(args.ablations_dir, "budget_granularity_sweep.csv"))
    fig_granularity_sweep(
        G, os.path.join(args.figures_dir, "fig_budget_granularity_sweep.png"))


if __name__ == "__main__":
    main()
