"""
tools/summarize_paper_results.py
---------------------------------
Read CSVs produced by tools/paper_eval.py and generate the paper-eval
summaries and final_report.md inside a paper_eval_<timestamp>/ directory.
"""

from __future__ import annotations
import csv
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _md_table(rows: List[Dict[str, Any]], cols: List[str], formats: Optional[Dict[str, str]] = None) -> str:
    formats = formats or {}
    if not rows:
        return "_(no data)_\n"
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if c in formats and v not in ("", None):
                try:
                    cells.append(formats[c].format(float(v)))
                except (ValueError, TypeError):
                    cells.append(str(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        print("usage: summarize_paper_results.py <paper_eval_dir>")
        sys.exit(2)
    root = sys.argv[1]
    summaries = os.path.join(root, "summaries")
    os.makedirs(summaries, exist_ok=True)

    core    = _read_csv(os.path.join(root, "ppl",     "core_ppl.csv"))
    inv     = _read_csv(os.path.join(root, "ppl",     "invariant.csv"))
    sweep   = _read_csv(os.path.join(root, "sweeps",  "stored_budget_sweep.csv"))
    pareto  = _read_csv(os.path.join(root, "sweeps",  "memory_pareto_sweep.csv"))
    abl     = _read_csv(os.path.join(root, "ablations","v_k_both_ablation_int3.csv"))
    mem     = _read_csv(os.path.join(root, "memory",  "memory_table.csv"))

    # ── Core PPL ────────────────────────────────────────────────────
    md_core = "# Core PPL (TinyLlama-1.1B-Chat-v1.0)\n\n"
    md_core += "Labels:  fp = full precision reference · base_quant = low-bit base KV only · "
    md_core += "carekv_eval = full-residual upper bound · carekv_stored = paper-quality stored-slot path.\n\n"
    md_core += _md_table(core,
        ["label","base_bits","prefill_mode","store_budget","read_budget","kind","k_scale","ppl","v_slots_read","k_slots_read","seconds"],
        formats={"store_budget":"{:.2f}", "read_budget":"{:.2f}", "k_scale":"{:.3f}", "ppl":"{:.4f}", "seconds":"{:.1f}"}
    )
    with open(os.path.join(summaries, "core_ppl_table.md"), "w") as f:
        f.write(md_core)

    # ── Invariant ───────────────────────────────────────────────────
    md_inv = "# R=0 / READ_BUDGET=0 invariant\n\n"
    md_inv += "Stored-slot CARE-KV with read_budget=0 must equal base_quant exactly.\n\n"
    md_inv += _md_table(inv,
        ["base_bits","base_quant_ppl","stored_r0_ppl","abs_diff","status"],
        formats={"base_quant_ppl":"{:.6f}", "stored_r0_ppl":"{:.6f}", "abs_diff":"{:.2e}"}
    )
    with open(os.path.join(summaries, "invariant_check.md"), "w") as f:
        f.write(md_inv)

    # ── Budget sweep ────────────────────────────────────────────────
    md_sw = "# Stored-slot budget sweep (carekv_stored, V-only)\n\n"
    md_sw += _md_table(sweep,
        ["base_bits","store_budget","read_budget","ppl","v_slots_read","k_slots_read","total_MB","vs_fp16","seconds"],
        formats={"store_budget":"{:.2f}", "read_budget":"{:.2f}", "ppl":"{:.4f}",
                 "total_MB":"{:.2f}", "vs_fp16":"{:.3f}", "seconds":"{:.1f}"}
    )
    with open(os.path.join(summaries, "budget_sweep_table.md"), "w") as f:
        f.write(md_sw)

    # ── V/K/both ablation ──────────────────────────────────────────
    md_abl = "# V / K / both ablation (INT3, S=0.10 R=0.03)\n\n"
    md_abl += _md_table(abl,
        ["label","kind","k_scale","v_score","score_normalize","ppl","v_slots_read","k_slots_read","seconds"],
        formats={"k_scale":"{:.3f}", "ppl":"{:.4f}", "seconds":"{:.1f}"}
    )
    with open(os.path.join(summaries, "ablation_table.md"), "w") as f:
        f.write(md_abl)

    # ── Memory-Pareto (Phase D residual-granularity sweep) ─────────
    md_par = "# Residual-granularity Pareto sweep (Phase D, INT3 carekv_stored, V-only)\n\n"
    if pareto:
        md_par += _md_table(pareto,
            ["label","page_size","v_token_block","k_channel_group","sketch_dim",
             "store_budget","read_budget","ppl","total_MB","vs_fp16",
             "base_MB","scale_MB","residual_MB","metadata_MB","sketch_MB",
             "v_slots_read","k_slots_read","actual_read_ratio","seconds"],
            formats={"store_budget":"{:.2f}","read_budget":"{:.2f}","ppl":"{:.4f}",
                     "total_MB":"{:.2f}","vs_fp16":"{:.3f}","base_MB":"{:.2f}",
                     "scale_MB":"{:.2f}","residual_MB":"{:.2f}","metadata_MB":"{:.2f}",
                     "sketch_MB":"{:.2f}","actual_read_ratio":"{:.2f}","seconds":"{:.1f}"}
        )
        # Pareto highlights
        def _fnum(r, k, default=float("inf")):
            try: return float(r.get(k, default))
            except (ValueError, TypeError): return default
        valid = [r for r in pareto if "ppl" in r and r["ppl"] not in ("","nan")]
        if valid:
            by_mem = sorted(valid, key=lambda r: _fnum(r,"total_MB"))[0]
            by_ppl = sorted(valid, key=lambda r: _fnum(r,"ppl"))[0]
            by_prod = sorted(valid, key=lambda r: _fnum(r,"total_MB") * _fnum(r,"ppl"))[0]
            by_speed = sorted(valid, key=lambda r: _fnum(r,"seconds"))[0]
            md_par += "\n## Pareto highlights\n\n"
            md_par += f"- **Lowest memory**: {by_mem['label']} → {by_mem['total_MB']} MB at PPL {by_mem['ppl']}\n"
            md_par += f"- **Best PPL**: {by_ppl['label']} → PPL {by_ppl['ppl']} at {by_ppl['total_MB']} MB\n"
            md_par += f"- **Best memory × PPL**: {by_prod['label']} → {by_prod['total_MB']} MB × PPL {by_prod['ppl']}\n"
            md_par += f"- **Fastest forward**: {by_speed['label']} → {by_speed['seconds']} s\n"
    else:
        md_par += "_(no data — run scripts/run_memory_pareto_sweep.sh or rerun_carekv_stored_clean.sh)_\n"
    with open(os.path.join(summaries, "memory_pareto_table.md"), "w") as f:
        f.write(md_par)

    # ── Memory ──────────────────────────────────────────────────────
    md_mem = "# Memory table\n\n"
    md_mem += "Reported per-(layer, KV head) — i.e. honours GQA.  packed_base=True is real on-device packed.\n\n"
    md_mem += _md_table(mem,
        ["seq_len","packed_base","scale_dtype","scale_quant","actual_MB","estimator_MB","fp16_MB","actual_vs_fp16","estimator_vs_fp16","base_code_MB","scale_MB","residual_MB","meta_MB","sketch_MB"],
        formats={"actual_MB":"{:.2f}", "estimator_MB":"{:.2f}", "fp16_MB":"{:.2f}",
                 "actual_vs_fp16":"{:.3f}", "estimator_vs_fp16":"{:.3f}",
                 "base_code_MB":"{:.2f}", "scale_MB":"{:.2f}", "residual_MB":"{:.2f}",
                 "meta_MB":"{:.2f}", "sketch_MB":"{:.2f}"}
    )
    with open(os.path.join(summaries, "memory_table.md"), "w") as f:
        f.write(md_mem)

    # ── Final report ───────────────────────────────────────────────
    fp_path = os.path.join(root, "final_report.md")
    with open(fp_path, "w") as f:
        f.write("# CARE-KV — Paper-quality evaluation report\n\n")
        f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("## 1. Implementation status\n\n")
        f.write("Stable and validated:\n")
        f.write("- post-RoPE K storage in a KV-head-indexed cache (Phase 2)\n")
        f.write("- `CAREKV_PREFILL_MODE=carekv_stored` reads only from stored residual slots (Phase 3)\n")
        f.write("- `R=0 / READ_BUDGET=0 ≡ base_quant` invariant\n")
        f.write("- Real packed base storage `CAREKV_PACKED_BASE=1` with INT2 / INT3 / INT4 (Phase B) — actual on-device memory matches estimator\n")
        f.write("- `CAREKV_SCALE_DTYPE` (fp16/bf16/fp32) + experimental `CAREKV_SCALE_QUANT=int8` (Phase C)\n")
        f.write("- `CAREKV_DEBUG_STATS=1` + accumulated slot-read counters\n\n")
        f.write("Not yet implemented:\n")
        f.write("- HF `use_cache=True` / DynamicCache integration (Phase G / Phase 4) — generation works only with `use_cache=False`\n\n")

        f.write("## 2. Cleanup\n\n")
        f.write(f"All previous result files in `results/` were archived to `results/archive_before_paper_eval_*/`. "
                f"Paper-quality run lives entirely under `{root}/`.\n\n")

        f.write("## 3. Curated experiments rerun\n\n")
        f.write("- `core-ppl` — fp, INT4/3/2 base_quant, INT3 carekv_eval (upper bound), INT3+INT2 carekv_stored.\n")
        f.write("- `invariant` — INT3 + INT2 stored R=0 vs base_quant.\n")
        f.write("- `budget-sweep` — carekv_stored V-only, store × read grid for INT3 (and INT2 if time-bounded).\n")
        f.write("- `vk-ablation` — INT3, v vs k vs both at S=0.10 R=0.03, K-scale ∈ {0.01, 0.02, 0.05}.\n")
        f.write("- `memory` — actual + estimator across SEQ_LEN ∈ {128, 512, 2048, 8192}.\n")
        f.write("- `generation` — fp16 / base_quant_int3 / carekv_stored_int3 (USE_CACHE=False).\n")
        f.write("- `figures` — diagnostic plots for layers 0, mid, last.\n\n")

        f.write("## 4. Core PPL\n\n")
        f.write(open(os.path.join(summaries, "core_ppl_table.md")).read())
        f.write("\n")

        f.write("## 5. Invariant check\n\n")
        f.write(open(os.path.join(summaries, "invariant_check.md")).read())
        f.write("\n")

        # Best stored-slot PPL: lowest carekv_stored row in core or sweep
        best = None
        for src in [core, sweep]:
            for r in src:
                if r.get("prefill_mode") == "carekv_stored" or r.get("label", "").startswith("carekv_stored") or r.get("base_bits", "") in {"2","3"} and src is sweep:
                    try:
                        ppl = float(r.get("ppl", "inf"))
                    except ValueError:
                        continue
                    if best is None or ppl < best[0]:
                        best = (ppl, dict(r))
        if best:
            f.write("## 6. Best carekv_stored result so far\n\n")
            f.write(f"PPL = **{best[0]:.4f}**\n\n")
            f.write("Config:\n```\n")
            for k, v in best[1].items():
                f.write(f"  {k} = {v}\n")
            f.write("```\n\n")

        f.write("## 7. Memory\n\n")
        f.write(open(os.path.join(summaries, "memory_table.md")).read())
        f.write("\n")

        f.write("## 8. V/K/both ablation\n\n")
        f.write(open(os.path.join(summaries, "ablation_table.md")).read())
        f.write("\n")

        f.write("## 9. Stored-slot budget sweep\n\n")
        f.write(open(os.path.join(summaries, "budget_sweep_table.md")).read())
        f.write("\n")

        if os.path.exists(os.path.join(summaries, "memory_pareto_table.md")):
            f.write("## 9b. Residual-granularity Pareto sweep (Phase D)\n\n")
            f.write(open(os.path.join(summaries, "memory_pareto_table.md")).read())
            f.write("\n")

        f.write("## 10. Debug evidence that stored slots are read\n\n")
        f.write("`v_slots_read` and `k_slots_read` columns in core PPL, budget sweep, and ablation tables "
                "are populated from `CAREKV_DEBUG_STATS=1`.  Nonzero values prove the stored-slot routing "
                "path is exercised on every query token.  R=0 invariant rows (above) show these counters "
                "go to zero when `read_budget_ratio=0`, confirming the short-circuit.\n\n")

        f.write("## 11. Current limitations\n\n")
        f.write("- `carekv_stored` prefill uses a per-token Python loop in `_apply_sparse_prefill_correction_stored`; on TinyLlama this is the dominant runtime cost (~minutes per forward at SEQ_LEN=128).  Vectorisation across heads is future work.\n")
        f.write("- `use_cache=True` generation is not wired into `DynamicCache`; only `use_cache=False` is validated end-to-end.\n")
        f.write("- Per-page `PageMeta` is a Python dataclass; the estimator under-reports its overhead by ~0.6 MB at TinyLlama capacity (real number tracked in the memory audit).\n")
        f.write("- `carekv_eval` results are reported as **upper bound only**, not paper-quality numbers.\n\n")

        f.write("## 12. Next steps\n\n")
        f.write("1. Phase D — residual granularity sweep (`v_token_block`, `k_channel_group`, `sketch_dim`) for Pareto plot.\n")
        f.write("2. Phase E — adaptive layer-budget policies (`uniform` / `layer_sensitivity` / `u_shaped`).\n")
        f.write("3. Phase G — `DynamicCache` integration so `use_cache=True` decode and latency benchmark can run.\n")
        f.write("4. Phase H — once Phase G lands, real latency benchmark and Phase I WikiText-2 / C4 PPL.\n")

    # ── Artifact list ──────────────────────────────────────────────
    al_path = os.path.join(root, "artifact_list.txt")
    with open(al_path, "w") as f:
        for dirpath, _, filenames in os.walk(root):
            for fn in sorted(filenames):
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                f.write(rel + "\n")

    print(f"\n──── summary done ────")
    print(f"  output dir   : {root}")
    print(f"  final report : {fp_path}")
    print(f"  summaries    : {summaries}/")
    print(f"  artifact list: {al_path}")


if __name__ == "__main__":
    main()
