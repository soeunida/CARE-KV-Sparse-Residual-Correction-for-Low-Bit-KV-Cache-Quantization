"""tools/make_budget_markdown.py

Render markdown tables for the budget audit:

  1. Inject the store-budget effective-budget table into
     summaries/budget_experiments_overview.md (at the <!-- STORE_SWEEP_TABLE -->
     marker), from ablations/store_budget_sweep.csv.
  2. Write summaries/budget_granularity_sweep.md from
     ablations/budget_granularity_sweep.csv.

DIAGNOSTIC artifacts (small synthetic forward-pass eval).
"""
from __future__ import annotations
import argparse, csv, os


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _g(r, k, default=""):
    return r.get(k, default)


def _store_table(rows):
    cols = [
        ("label", "label"),
        ("req_SK", "req_SK"), ("eff_SK", "eff_SK"), ("K_cand_cap", "K_cap"),
        ("req_SV", "req_SV"), ("eff_SV", "eff_SV"), ("V_cand_cap", "V_cap"),
        ("budget_wasted", "wasted"), ("store_util", "util"),
        ("ppl", "PPL"), ("K_reads", "K_reads"), ("V_reads", "V_reads"),
        ("recovered_K_elems", "recK"), ("recovered_V_elems", "recV"),
        ("residual_mem_MB", "res_MB"),
    ]
    head = "| " + " | ".join(h for _, h in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    out = [head, sep]
    for r in rows:
        if r.get("error"):
            continue
        out.append("| " + " | ".join(str(_g(r, c)) for c, _ in cols) + " |")
    return "\n".join(out)


def _inject_store_table(overview_path, store_rows):
    marker = "<!-- STORE_SWEEP_TABLE -->"
    if not os.path.exists(overview_path) or not store_rows:
        print(f"[md] skip store-table inject (overview or rows missing)")
        return
    text = open(overview_path).read()
    table = _store_table(store_rows)
    if marker in text:
        text = text.replace(marker, table)
    else:
        text += "\n\n" + table + "\n"
    with open(overview_path, "w") as f:
        f.write(text)
    print(f"[md] injected store-budget table -> {overview_path}")


def _granularity_md(rows):
    lines = []
    lines.append("# Residual granularity sensitivity sweep (diagnostic)")
    lines.append("")
    lines.append("**Status: diagnostic.** Small synthetic forward-pass eval "
                 "(TinyLlama-1.1B, SEQ_LEN=128, INT3 base, `route_policy=joint`, "
                 "`score_normalize=1`, `correction_impl=cached`). Validates the "
                 "candidate-cap interpretation of the store budget; does **not** "
                 "by itself establish a new paper-best granularity.")
    lines.append("")
    lines.append("The store budget selects from per-page candidate pools whose "
                 "size is fixed by granularity:")
    lines.append("")
    lines.append("```")
    lines.append("K_cand_cap = head_dim      / k_channel_group   (head_dim=64)")
    lines.append("V_cand_cap = ceil(page_size / v_token_block)   (page_size=16)")
    lines.append("```")
    lines.append("")
    lines.append("Each cell stores ALL candidates at its granularity "
                 "(`SK=K_cap`, `SV=V_cap` — the minimum-storage equivalent for "
                 "that granularity) and reads at the paper budget "
                 "(`RK=min(2,K_cap)`, `RV=min(2,V_cap)`).")
    lines.append("")
    cols = [
        ("k_channel_group", "kcg"), ("v_token_block", "vtb"),
        ("K_cand_cap", "K_cap"), ("V_cand_cap", "V_cap"),
        ("eff_SK", "SK"), ("eff_SV", "SV"),
        ("read_abs_k", "RK"), ("read_abs_v", "RV"),
        ("ppl", "PPL"),
        ("K_reads", "K_reads"), ("V_reads", "V_reads"),
        ("recovered_K_elems", "recK"), ("recovered_V_elems", "recV"),
        ("store_util", "util"),
        ("residual_mem_MB", "res_MB"), ("cache_mem_MB", "cache_MB"),
        ("seconds", "sec"),
    ]
    lines.append("| " + " | ".join(h for _, h in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    valid = [r for r in rows if not r.get("error")]
    for r in valid:
        lines.append("| " + " | ".join(str(_g(r, c)) for c, _ in cols) + " |")
    lines.append("")
    # Headline: best PPL cell + paper cell.
    def _f(r, k):
        try:
            return float(r.get(k, "inf"))
        except ValueError:
            return float("inf")
    if valid:
        best = min(valid, key=lambda r: _f(r, "ppl"))
        paper = next((r for r in valid
                      if str(_g(r, "k_channel_group")) == "32"
                      and str(_g(r, "v_token_block")) == "4"), None)
        lines.append(f"**Best PPL:** `kcg={_g(best,'k_channel_group')}, "
                     f"vtb={_g(best,'v_token_block')}` "
                     f"(K_cap={_g(best,'K_cand_cap')}, V_cap={_g(best,'V_cand_cap')}) "
                     f"→ PPL {_g(best,'ppl')}, residual {_g(best,'residual_mem_MB')} MB.")
        if paper is not None:
            lines.append("")
            lines.append(f"**Paper-best granularity** (`kcg=32, vtb=4` → caps 2×4): "
                         f"PPL {_g(paper,'ppl')}, residual {_g(paper,'residual_mem_MB')} MB.")
        lines.append("")
        lines.append("**Interpretation.** Finer granularity raises the candidate "
                     "caps (more, smaller residuals) and recovered elements at a "
                     "residual-memory cost. Whether the PPL gain justifies the "
                     "memory is the decision gate (§5 of `CLAUDE.md`); the locked "
                     "paper-best config is unchanged unless a granularity clearly "
                     "improves PPL without inflating memory too much.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", required=True)
    args = ap.parse_args()
    abl = os.path.join(args.paper_dir, "ablations")
    summ = os.path.join(args.paper_dir, "summaries")

    store_rows = _read(os.path.join(abl, "store_budget_sweep.csv"))
    _inject_store_table(os.path.join(summ, "budget_experiments_overview.md"),
                        store_rows)

    gran_rows = _read(os.path.join(abl, "budget_granularity_sweep.csv"))
    if gran_rows:
        out_md = os.path.join(summ, "budget_granularity_sweep.md")
        with open(out_md, "w") as f:
            f.write(_granularity_md(gran_rows))
        print(f"[md] wrote {out_md}")
    else:
        print("[md] no granularity rows; skipped budget_granularity_sweep.md")


if __name__ == "__main__":
    main()
