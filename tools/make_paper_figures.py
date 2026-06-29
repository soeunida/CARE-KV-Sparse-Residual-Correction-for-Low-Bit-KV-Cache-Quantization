"""tools/make_paper_figures.py

Phase K — generate paper-quality summary figures from the CSVs under a
paper_eval_<ts> directory.  Reads each CSV defensively (skips missing
files) and writes one PNG per figure into `<paper_dir>/figures/`.

Usage:
  PYTHONPATH=/home/soeun python tools/make_paper_figures.py \
      --paper-dir results/paper_eval_20260529_015053
"""
from __future__ import annotations
import argparse, csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: str):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[figures] saved {out_path}", flush=True)


def fig_wikitext2_ppl(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/ppl_dataset/wikitext2_paper_ppl.csv")
    if not rows:
        print("[figures] skip wikitext2 (no CSV)"); return
    order = ["fp16", "base_quant_int4", "base_quant_int3", "base_quant_int2",
             "carekv_stored_int3_optimized"]
    pretty = {"fp16": "fp16",
              "base_quant_int4": "INT4\nbase_quant",
              "base_quant_int3": "INT3\nbase_quant",
              "base_quant_int2": "INT2\nbase_quant",
              "carekv_stored_int3_optimized": "INT3\nCARE-KV"}
    colors = {"fp16": "#1f77b4",
              "base_quant_int4": "#aec7e8",
              "base_quant_int3": "#ff7f0e",
              "base_quant_int2": "#d62728",
              "carekv_stored_int3_optimized": "#2ca02c"}
    by_mode = {r["mode"]: float(r["ppl"]) for r in rows}
    xs = [m for m in order if m in by_mode]
    ys = [by_mode[m] for m in xs]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar([pretty[m] for m in xs], ys, color=[colors[m] for m in xs])
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y, f"{y:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("WikiText-2 PPL (log scale)")
    ax.set_title("WikiText-2 PPL on TinyLlama-1.1B (N=16, SL=128)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, f"{paper_dir}/figures/fig_wikitext2_ppl.png")


def fig_memory_pareto(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/sweeps/memory_pareto_sweep.csv")
    if not rows:
        print("[figures] skip memory_pareto (no CSV)"); return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        mem = float(r["total_MB"]); ppl = float(r["ppl"])
        label = r["label"]
        is_fp = "fp16" in label.lower()
        is_base = "base_quant" in label.lower() and "carekv" not in label.lower()
        color = "#1f77b4" if is_fp else ("#ff7f0e" if is_base else "#2ca02c")
        marker = "*" if is_fp else ("s" if is_base else "o")
        size = 180 if is_fp else (90 if is_base else 80)
        ax.scatter(mem, ppl, s=size, c=color, marker=marker,
                   edgecolors="black", linewidths=0.5, label=label)
        ax.annotate(label, (mem, ppl), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Total KV-cache memory (MB)")
    ax.set_ylabel("PPL")
    ax.set_title("Memory–PPL Pareto: CARE-KV vs base_quant vs fp16")
    ax.grid(linestyle="--", alpha=0.4)
    _save(fig, f"{paper_dir}/figures/fig_memory_pareto.png")


def fig_absolute_budget_sweep(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/sweeps/absolute_budget_sweep.csv")
    if not rows:
        print("[figures] skip absolute_budget_sweep (no CSV)"); return
    rows.sort(key=lambda r: (int(r["read_abs_k"]) + int(r["read_abs_v"]),
                              int(r["store_abs_k"]) + int(r["store_abs_v"])))
    xs = list(range(len(rows)))
    ys = [float(r["ppl"]) for r in rows]
    labels = [f"S(k{r['store_abs_k']},v{r['store_abs_v']}) "
              f"R(k{r['read_abs_k']},v{r['read_abs_v']})" for r in rows]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bars = ax.bar(xs, ys, color="#2ca02c")
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y, f"{y:.2f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("PPL"); ax.set_title("Absolute K/V budget sweep (INT3, TinyLlama)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, f"{paper_dir}/figures/fig_absolute_budget_sweep.png")


def fig_route_policies(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/ablations/route_policies_int3.csv")
    if not rows:
        print("[figures] skip route_policies (no CSV)"); return
    kinds = ["v", "k", "both"]
    policies = sorted({r["route_policy"] for r in rows})
    grid = {(r["kind"], r["route_policy"]): float(r["ppl_cached"]) for r in rows}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(policies)
    xs = list(range(len(kinds)))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, pol in enumerate(policies):
        ys = [grid.get((k, pol), float("nan")) for k in kinds]
        ax.bar([x + (i - (len(policies)-1)/2) * width for x in xs], ys,
               width=width, label=pol, color=palette[i % len(palette)])
    ax.set_xticks(xs); ax.set_xticklabels(kinds)
    ax.set_xlabel("residual kind"); ax.set_ylabel("PPL (cached impl)")
    ax.set_title("Routing policy ablation (INT3, TinyLlama, synthetic SL=64)")
    ax.legend(title="route_policy", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, f"{paper_dir}/figures/fig_route_policies.png")


def fig_vk_both_ablation(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/ablations/v_k_both_ablation_int3.csv")
    if not rows:
        print("[figures] skip vk_both (no CSV)"); return
    labels = [r["label"] for r in rows]
    ys = [float(r["ppl"]) for r in rows]
    xs = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(xs, ys, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                                   "#9467bd", "#8c564b", "#e377c2"][:len(xs)])
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y, f"{y:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("PPL"); ax.set_title("V / K / Both ablation (INT3, TinyLlama)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, f"{paper_dir}/figures/fig_vk_both_ablation.png")


def fig_layer_budget_policy(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/ablations/layer_budget_policy.csv")
    if not rows:
        print("[figures] skip layer_budget_policy (no CSV)"); return
    xs = [r["policy"] for r in rows]
    ys = [float(r["ppl"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(xs, ys, color="#9467bd")
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, y, f"{y:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("PPL"); ax.set_title("Layer-wise budget policy (INT3, TinyLlama)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, f"{paper_dir}/figures/fig_layer_budget_policy.png")


def fig_long_context_retrieval(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/long_context/long_context_retrieval.csv")
    if not rows:
        print("[figures] skip long_context (no CSV)"); return
    tasks = ["kv_retrieval", "boundary", "copy"]
    modes = ["fp16", "base_quant_int3", "carekv_int3_both"]
    pretty_mode = {"fp16": "fp16", "base_quant_int3": "INT3 base_quant",
                   "carekv_int3_both": "INT3 CARE-KV"}
    grid = {(r["task"], r["label"]): (float(r["exact_match"]),
                                       float(r["char_acc"])) for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.8 / len(modes)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    xs = list(range(len(tasks)))
    for col, (metric_idx, metric_name) in enumerate([(0, "Exact match"),
                                                       (1, "Character accuracy")]):
        ax = axes[col]
        for i, mode in enumerate(modes):
            ys = [grid.get((t, mode), (float("nan"), float("nan")))[metric_idx]
                  for t in tasks]
            ax.bar([x + (i - (len(modes)-1)/2) * width for x in xs], ys,
                   width=width, label=pretty_mode[mode], color=palette[i])
        ax.set_xticks(xs); ax.set_xticklabels(tasks)
        ax.set_ylim(0, 1.05); ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Long-context retrieval (TinyLlama, SL=128, n_pairs=6, 5 trials)",
                 fontsize=11)
    _save(fig, f"{paper_dir}/figures/fig_long_context_retrieval.png")


def fig_multimodel_ppl(paper_dir: str):
    rows = _read_csv(f"{paper_dir}/ppl_dataset/multimodel_wikitext2.csv")
    if not rows:
        print("[figures] skip multimodel (no CSV)"); return
    models = []
    for r in rows:
        m = r["model"]
        if m not in models:
            models.append(m)
    modes = ["fp16", "base_quant_int3", "carekv_stored_int3_optimized"]
    pretty_mode = {"fp16": "fp16", "base_quant_int3": "INT3 base_quant",
                   "carekv_stored_int3_optimized": "INT3 CARE-KV"}
    grid = {(r["model"], r["mode"]): float(r["ppl"]) for r in rows}
    short = {m: m.split("/")[-1] for m in models}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(modes)
    xs = list(range(len(models)))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, mode in enumerate(modes):
        ys = [grid.get((m, mode), float("nan")) for m in models]
        bars = ax.bar([x + (i - (len(modes)-1)/2) * width for x in xs], ys,
                      width=width, label=pretty_mode[mode], color=palette[i])
        for b, y in zip(bars, ys):
            if y == y:
                ax.text(b.get_x() + b.get_width()/2, y, f"{y:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs); ax.set_xticklabels([short[m] for m in models], fontsize=9)
    ax.set_ylabel("WikiText-2 PPL")
    ax.set_title("Multi-model WikiText-2 PPL (SL=128, N=4 smoke)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    _save(fig, f"{paper_dir}/figures/fig_multimodel_ppl.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", required=True)
    args = ap.parse_args()
    paper_dir = args.paper_dir
    fig_wikitext2_ppl(paper_dir)
    fig_memory_pareto(paper_dir)
    fig_absolute_budget_sweep(paper_dir)
    fig_route_policies(paper_dir)
    fig_vk_both_ablation(paper_dir)
    fig_layer_budget_policy(paper_dir)
    fig_long_context_retrieval(paper_dir)
    fig_multimodel_ppl(paper_dir)


if __name__ == "__main__":
    main()
