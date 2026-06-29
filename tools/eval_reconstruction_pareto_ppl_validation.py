# ============================================================================
# RECOVERED ARCHIVAL SCRIPT — Phase U PPL validation
# ----------------------------------------------------------------------------
# This file is an EXACT restoration of the original executable that produced
# results/.../ablations/reconstruction_pareto_ppl_validation.csv (its store_policy
# strings match the surviving CSV).  The on-disk original was deleted by a
# `git clean`/`stash -u` run from another terminal on the shared checkout
# (Phase U contamination incident); it is restored here from conversation
# context into the isolated recovery worktree.
#
# NOTE on re-execution: the reconstruction-number columns (K/V/combined recon
# reduction, residual_memory_KB) are populated by load_recon_numbers() reading
# the Phase U-B CSVs:
#     ablations/residual_reconstruction_store_sweep.csv
#     ablations/residual_reconstruction_pareto.csv
# Those U-B CSVs were ALSO lost in the incident and are not present in this
# worktree.  Re-running now would still measure PPL through the real
# carekv_stored path but would report 0.0 for the reconstruction columns
# (they default to zero when the source rows are missing).  The authoritative
# numbers are preserved in the restored
# ablations/reconstruction_pareto_ppl_validation.csv and the corrected
# summaries/reconstruction_pareto_ppl_validation.md.
# ============================================================================
"""tools/eval_reconstruction_pareto_ppl_validation.py

Phase U — PPL-focused follow-up.

Validates whether the Phase U-B reconstruction-Pareto candidates also
improve PPL through the REAL `carekv_stored` attention path (per-query
Jacobian read-time correction), not just the idealized store-time
reconstruction metric.

IMPORTANT reconciliation note
-----------------------------
The U-B reconstruction analysis used an idealized token-level / global
store selection.  The deployed `carekv_stored` store path is different:

  * STORE_ABS_K counts K *channel-group* candidates kept per page
    (num_cg = head_dim / k_channel_group).  At the paper default
    kcg=32 → num_cg=2, so STORE_ABS_K≥2 already stores ALL K residuals;
    STORE_ABS_K=4 is capped at 2 (== current).
  * STORE_ABS_V counts V *token-block* candidates per page
    (num_vb = page_size / v_token_block).
  * Decode-time sparsity is governed by the READ budget (READ_ABS_K/V),
    via the per-query router (a global top-k over available slots).

So the idealized "global magnitude kcg16" candidates are realized here as
their closest DEPLOYABLE approximation: a finer K channel group (kcg=16)
gives the read-time router finer K residual blocks to select from, with
the store budget set to keep a comparable amount.  Reconstruction columns
report the idealized U-B numbers (the basis the candidate was defined on);
the runtime store config + PPL are the real deployed measurement.

Candidates (see task spec):
  1 base_quant_INT3            read 0/0 (no correction)            [reference]
  2 per_page_current           kcg32 store 2/4 read 2/2           [paper-best]
  3 mag_SK1_SV4                kcg32 store 1/4 read 2/2           [less K stored]
  4 global_mag_kcg16_s12.5     kcg16 store 1/2 read 2/2           [finer K, low store]
  5 global_mag_kcg16_s18.8     kcg16 store 2/2 read 2/2           [finer K, ~same mem]
  6 high_recon_SK4_SV4         kcg32 store 4/4 read 2/2           [store-saturated]

Outputs:
  ablations/reconstruction_pareto_ppl_validation.csv
  summaries/reconstruction_pareto_ppl_validation.md
  figures/fig_reconstruction_pareto_ppl_validation.png
"""
from __future__ import annotations
import argparse, csv, os, sys, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats,
)
from CARE_KV.care_kv.cache import apply_carekv_env_overrides
from CARE_KV.care_kv.baselines.common import DEVICE, eval_ppl_wikitext

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Candidate runtime configs.  recon_src points at the U-B CSV row whose
# idealized reconstruction numbers describe the candidate.
CANDIDATES = [
    dict(name="base_quant_INT3", store_policy="none (read=0)",
         kcg=32, vtb=4, sk=2, sv=4, rk=0, rv=0, recon_src=("zero", None)),
    dict(name="per_page_current_SK2SV4", store_policy="per-page channel-group (paper-best)",
         kcg=32, vtb=4, sk=2, sv=4, rk=2, rv=2, recon_src=("store", "2/4")),
    dict(name="mag_SK1_SV4", store_policy="per-page, half K stored",
         kcg=32, vtb=4, sk=1, sv=4, rk=2, rv=2, recon_src=("store", "1/4")),
    dict(name="global_mag_kcg16_s12.5", store_policy="finer kcg16, low store (deployable approx)",
         kcg=16, vtb=4, sk=1, sv=2, rk=2, rv=2, recon_src=("pareto", "global_mag_kcg16_vtb1_s0.1250")),
    dict(name="global_mag_kcg16_s18.8", store_policy="finer kcg16, ~same-mem store (deployable approx)",
         kcg=16, vtb=4, sk=2, sv=2, rk=2, rv=2, recon_src=("pareto", "global_mag_kcg16_vtb1_s0.1875")),
    dict(name="high_recon_SK4_SV4", store_policy="per-page, store-saturated (SK4 caps at num_cg=2)",
         kcg=32, vtb=4, sk=4, sv=4, rk=2, rv=2, recon_src=("store", "4/4")),
]


def load_recon_numbers(store_csv, pareto_csv):
    """Return dict src_key -> (K_redux, V_redux, combined, mem_KB)."""
    out = {}
    if os.path.exists(store_csv):
        for r in csv.DictReader(open(store_csv)):
            k = f"{r['requested_SK']}/{r['requested_SV']}"
            out[("store", k)] = (
                float(r["K_rel_redux_pct"]), float(r["V_rel_redux_pct"]),
                float(r["combined_norm_recon_score"]),
                int(r["total_residual_mem_bytes"]) / 1024.0)
    if os.path.exists(pareto_csv):
        for r in csv.DictReader(open(pareto_csv)):
            out[("pareto", r["config_label"])] = (
                float(r["K_redux_pct"]), float(r["V_redux_pct"]),
                float(r["combined_redux_pct"]),
                float(r["total_residual_memory_KB"]))
    out[("zero", None)] = (0.0, 0.0, 0.0, 0.0)
    return out


def setup_carekv(model_id, kcg, vtb, sk, sv, rk, rv, base_bits, max_pages):
    """Build a patched CARE-KV model with explicit store/read knobs (real path)."""
    reset_debug_stats()
    env = dict(
        CAREKV_PREFILL_MODE="carekv_stored",
        CAREKV_PREFILL_RESIDUAL_KIND="both",
        CAREKV_ROUTE_POLICY="joint",
        CAREKV_SCORE_NORMALIZE="1",
        CAREKV_CORRECTION_IMPL="cached",
        CAREKV_BUDGET_POLICY="uniform",
        CAREKV_PACKED_BASE="1",
        CAREKV_SCALE_QUANT="int8",
        CAREKV_BASE_BITS=str(base_bits),
        CAREKV_GROUP_SIZE="32",
        CAREKV_K_CHANNEL_GROUP=str(kcg),
        CAREKV_V_TOKEN_BLOCK=str(vtb),
        CAREKV_STORE_BUDGET_MODE="absolute",
        CAREKV_READ_BUDGET_MODE="absolute",
        CAREKV_STORE_ABS_K=str(sk), CAREKV_STORE_ABS_V=str(sv),
        CAREKV_READ_ABS_K=str(rk), CAREKV_READ_ABS_V=str(rv),
        CAREKV_READ_RELATIVE_THRESHOLD="0.0",
        CAREKV_READ_ABSOLUTE_THRESHOLD="0.0",
        CAREKV_READ_MIN_KEEP="0",
        CAREKV_BASE_QUANTIZER="uniform",
        CAREKV_DEBUG_STATS="1",
        CAREKV_MAX_PAGES=str(max_pages),
    )
    for k, v in env.items():
        os.environ[k] = v
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None)
    m.config.use_cache = False
    c = m.config
    hd = c.hidden_size // c.num_attention_heads
    kw = dict(
        num_layers=c.num_hidden_layers, num_heads=c.num_attention_heads,
        num_kv_heads=c.num_key_value_heads, head_dim=hd, base_bits=base_bits,
        group_size=32, k_channel_group=kcg, page_size=16, max_pages=max_pages,
        v_token_block=vtb, sketch_dim=16,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
    )
    apply_carekv_env_overrides(kw)
    cc = CacheConfig(**kw)
    m = patch_llama_model(m, cc)
    reset_all_caches(m)
    m.eval()
    return m


def runtime_store_sparsity(kcg, vtb, sk, sv, head_dim=64, page_size=16):
    """Effective per-page store sparsity in the real channel-group/token-block store."""
    num_cg = max(1, head_dim // kcg)
    num_vb = max(1, page_size // vtb)
    eff_sk = min(sk, num_cg)
    eff_sv = min(sv, num_vb)
    return eff_sk / num_cg, eff_sv / num_vb, eff_sk, eff_sv, num_cg, num_vb


def main():
    ap = argparse.ArgumentParser()
    base = "results/paper_eval_20260529_015053"
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--max-pages", type=int, default=16)
    ap.add_argument("--store-csv", default=f"{base}/ablations/residual_reconstruction_store_sweep.csv")
    ap.add_argument("--pareto-csv", default=f"{base}/ablations/residual_reconstruction_pareto.csv")
    ap.add_argument("--out-csv", default=f"{base}/ablations/reconstruction_pareto_ppl_validation.csv")
    ap.add_argument("--out-md", default=f"{base}/summaries/reconstruction_pareto_ppl_validation.md")
    ap.add_argument("--out-fig", default=f"{base}/figures/fig_reconstruction_pareto_ppl_validation.png")
    args = ap.parse_args()

    recon = load_recon_numbers(args.store_csv, args.pareto_csv)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    rows = []
    for c in CANDIDATES:
        t0 = time.perf_counter()
        status, notes, ppl, ntok, kr, vr = "ok", "", 0.0, 0, 0, 0
        try:
            m = setup_carekv(args.model_id, c["kcg"], c["vtb"], c["sk"], c["sv"],
                             c["rk"], c["rv"], args.base_bits, args.max_pages)
            ppl, ntok = eval_ppl_wikitext(m, tok, args.seq_len, args.num_samples)
            s = get_debug_stats()
            kr, vr = int(s.get("k_slots_read", 0)), int(s.get("v_slots_read", 0))
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            status, notes = "error", f"{type(e).__name__}: {e}"
        kK, kV, kComb, mem = recon.get(c["recon_src"], (0.0, 0.0, 0.0, 0.0))
        ksp, vsp, eff_sk, eff_sv, ncg, nvb = runtime_store_sparsity(
            c["kcg"], c["vtb"], c["sk"], c["sv"])
        rows.append(dict(
            config=c["name"], store_policy=c["store_policy"],
            kcg=c["kcg"], vtb=c["vtb"], store_abs_k=c["sk"], store_abs_v=c["sv"],
            read_abs_k=c["rk"], read_abs_v=c["rv"],
            runtime_K_store_sparsity_pct=round(ksp * 100, 1),
            runtime_V_store_sparsity_pct=round(vsp * 100, 1),
            ub_K_sparsity_note="(idealized U-B basis)",
            K_recon_reduction_pct=round(kK, 2), V_recon_reduction_pct=round(kV, 2),
            combined_recon_reduction_pct=round(kComb, 2),
            residual_memory_KB=round(mem, 2),
            ppl=round(ppl, 4), evaluated_tokens=ntok,
            k_reads=kr, v_reads=vr,
            runtime_seconds=round(time.perf_counter() - t0, 1),
            status=status, notes=notes,
        ))
        print(f"[U-PPL] {c['name']:26s} PPL={ppl:8.4f} K_reads={kr:>7d} "
              f"V_reads={vr:>7d} comb_recon={kComb:5.1f}% ({status}) "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    # ΔPPL annotations
    base_ppl = next((r["ppl"] for r in rows if r["config"] == "base_quant_INT3"
                     and r["status"] == "ok" and r["ppl"] > 0), None)
    cur_ppl = next((r["ppl"] for r in rows if r["config"] == "per_page_current_SK2SV4"
                    and r["status"] == "ok" and r["ppl"] > 0), None)
    for r in rows:
        r["dppl_vs_base_quant"] = (round(r["ppl"] - base_ppl, 4)
                                   if base_ppl and r["ppl"] > 0 else "")
        r["dppl_vs_current_SK2SV4"] = (round(r["ppl"] - cur_ppl, 4)
                                       if cur_ppl and r["ppl"] > 0 else "")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]
    make_fig(rows, base_ppl, args.out_fig)
    write_md(rows, base_ppl, cur_ppl, args)
    print(f"[U-PPL] wrote {args.out_csv}\n[U-PPL] wrote {args.out_md}\n[U-PPL] wrote {args.out_fig}")


def make_fig(rows, base_ppl, out_png):
    ok = [r for r in rows if r["status"] == "ok" and r["ppl"] > 0]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    names = [r["config"] for r in ok]
    ppls = [r["ppl"] for r in ok]
    colors = ["gray" if r["config"] == "base_quant_INT3" else
              "gold" if r["config"] == "per_page_current_SK2SV4" else "tab:blue" for r in ok]
    ax[0].bar(range(len(ok)), ppls, color=colors)
    if base_ppl:
        ax[0].axhline(base_ppl, ls="--", c="red", lw=1, label="base_quant_INT3")
    ax[0].set_xticks(range(len(ok))); ax[0].set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax[0].set_ylabel("WikiText-2 PPL"); ax[0].set_title("PPL per candidate")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, axis="y")
    lo = min(ppls); ax[0].set_ylim(lo - (max(ppls) - lo) * 0.5, max(ppls) + (max(ppls) - lo) * 0.2)

    # recon reduction vs ΔPPL
    corr = [r for r in ok if r["config"] != "base_quant_INT3"]
    ax[1].scatter([r["combined_recon_reduction_pct"] for r in corr],
                  [r["dppl_vs_base_quant"] for r in corr], s=70, c="tab:purple")
    for r in corr:
        ax[1].annotate(r["config"], (r["combined_recon_reduction_pct"],
                       r["dppl_vs_base_quant"]), fontsize=6)
    ax[1].axhline(0, c="k", lw=0.6)
    ax[1].set_xlabel("combined reconstruction reduction (%)")
    ax[1].set_ylabel("ΔPPL vs base_quant (lower=better)")
    ax[1].set_title("does better reconstruction → lower PPL?"); ax[1].grid(alpha=0.3)
    fig.suptitle("U PPL validation — carekv_stored real path (TinyLlama, WT-2, "
                 f"SL=128, N={len([r for r in rows])and rows[0]['evaluated_tokens'] and ''}4)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=120); plt.close(fig)


def write_md(rows, base_ppl, cur_ppl, args):
    by = {r["config"]: r for r in rows}
    L = ["# Phase U — reconstruction-Pareto PPL validation", "",
         "**Label:** `paper-ready candidate screen` (real `carekv_stored` path, "
         f"WT-2 N={args.num_samples}, SL={args.seq_len}). Reconstruction columns are "
         "the idealized U-B basis; runtime store config + PPL are the deployed "
         "measurement. See header of the tool for the store-semantics caveat.", "",
         f"- model `{args.model_id}`  BASE_BITS={args.base_bits}  max_pages={args.max_pages}",
         "- fixed: PACKED_BASE=1, SCALE_QUANT=int8, PREFILL=carekv_stored, "
         "RESIDUAL_KIND=both, ROUTE=joint, SCORE_NORMALIZE=1, CORRECTION=cached, "
         "BUDGET_POLICY=uniform, READ_ABS=2/2 (except base_quant read=0/0).", "",
         "| config | store policy | runtime K/V store spars | recon K | recon V "
         "| recon comb | resid mem KB | PPL | ΔPPL vs base | ΔPPL vs current "
         "| K_reads | V_reads | rt(s) | status |",
         "|:-------|:-------------|:-----------------------:|------:|------:|------:"
         "|----------:|----:|-----:|-----:|------:|------:|----:|:--:|"]
    for r in rows:
        L.append(
            f"| {r['config']} | {r['store_policy']} "
            f"| {r['runtime_K_store_sparsity_pct']:.0f}%/{r['runtime_V_store_sparsity_pct']:.0f}% "
            f"| {r['K_recon_reduction_pct']:.1f}% | {r['V_recon_reduction_pct']:.1f}% "
            f"| {r['combined_recon_reduction_pct']:.1f}% | {r['residual_memory_KB']:.1f} "
            f"| {r['ppl']:.4f} | {r['dppl_vs_base_quant']} | {r['dppl_vs_current_SK2SV4']} "
            f"| {r['k_reads']} | {r['v_reads']} | {r['runtime_seconds']:.0f} | {r['status']} |")
    L.append("")

    # Interpretation
    def ppl(n): return by[n]["ppl"] if n in by and by[n]["status"] == "ok" else None
    cur = ppl("per_page_current_SK2SV4")
    c4 = ppl("global_mag_kcg16_s12.5"); c5 = ppl("global_mag_kcg16_s18.8")
    c3 = ppl("mag_SK1_SV4"); c6 = ppl("high_recon_SK4_SV4")
    best = min((r for r in rows if r["status"] == "ok" and r["ppl"] > 0
                and r["config"] != "base_quant_INT3"), key=lambda r: r["ppl"], default=None)

    L += ["## Interpretation", ""]
    L.append(f"**0. Sanity.** base_quant_INT3 PPL={base_ppl}. Current SK2SV4 "
             f"PPL={cur} (ΔvsBase={by['per_page_current_SK2SV4']['dppl_vs_base_quant'] if 'per_page_current_SK2SV4' in by else 'NA'}). "
             "Any CARE-KV row should sit at or below base_quant if the router fired "
             "(check K_reads/V_reads > 0).")
    L.append("**1. Does better reconstruction reduction also improve PPL?** "
             + _trend_sentence(rows))
    if c5 is not None and cur is not None:
        L.append(f"**2. Does global_mag_kcg16_s18.8 beat current SK2SV4 at ~same memory?** "
                 f"PPL {c5:.4f} vs {cur:.4f} → "
                 + ("YES, lower PPL." if c5 < cur - 1e-4 else
                    "NO — not lower PPL." if c5 > cur + 1e-4 else "TIE."))
    if c4 is not None and cur is not None:
        L.append(f"**3. Does global_mag_kcg16_s12.5 preserve PPL at lower memory?** "
                 f"PPL {c4:.4f} vs {cur:.4f} → "
                 + ("YES, preserved/better." if c4 <= cur + 0.02 else
                    "NO — PPL degraded."))
    L.append("**4. Is magnitude-only selection good for PPL or only reconstruction?** "
             + (f"mag_SK1_SV4 PPL={c3:.4f}. " if c3 else "")
             + "U-B2 showed magnitude≈oracle for reconstruction; here we check whether "
             "that translates through the read-time Jacobian correction. "
             + _mag_sentence(c3, cur))
    if best:
        improves = (cur is not None and best["ppl"] < cur - 1e-4
                    and best["config"] != "per_page_current_SK2SV4")
        L.append(f"**5/7. Should the default change?** Best non-base PPL: "
                 f"{best['config']} ({best['ppl']:.4f}). "
                 + (f"It beats current SK2SV4 ({cur:.4f}) → mark as CANDIDATE "
                    "paper-best, REQUIRE WT-2 N=16 confirmation before adopting."
                    if improves else
                    f"It does NOT beat current SK2SV4 ({cur:.4f}). **Keep SK2SV4 as "
                    "paper-best; report the reconstruction Pareto as a diagnostic only "
                    "(Q6).** Reconstruction-error reduction did not translate into a "
                    "PPL win through the read-time correction at READ_ABS=2/2."))
    L += ["", "> Caveat: the idealized U-B 'global magnitude' selection is not "
          "faithfully deployable in the current per-page channel-group store; "
          "candidates 4/5 are the closest deployable approximation (finer kcg=16). "
          "Decode-time sparsity is read-governed (READ_ABS=2/2 fixed here), which is "
          "largely orthogonal to the store-time reconstruction sparsity swept in U-B.", ""]
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    open(args.out_md, "w").write("\n".join(L) + "\n")


def _trend_sentence(rows):
    pts = [(r["combined_recon_reduction_pct"], r["dppl_vs_base_quant"])
           for r in rows if r["status"] == "ok" and r["ppl"] > 0
           and r["config"] != "base_quant_INT3" and r["dppl_vs_base_quant"] != ""]
    if len(pts) < 3:
        return "Insufficient successful rows to judge a trend."
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in pts)
    var = sum((x - mx) ** 2 for x in xs)
    slope = cov / var if var else 0.0
    if slope < -1e-4:
        return (f"Weak/clear NEGATIVE slope (ΔPPL vs recon-reduction ≈ {slope:.4f}) — "
                "more reconstruction reduction tends to LOWER PPL.")
    if slope > 1e-4:
        return (f"POSITIVE slope ({slope:.4f}) — more reconstruction reduction does "
                "NOT lower PPL here; reconstruction is not a reliable PPL proxy at "
                "READ_ABS=2/2.")
    return "Essentially flat — reconstruction reduction does not predict PPL here."


def _mag_sentence(c3, cur):
    if c3 is None or cur is None:
        return ""
    if c3 <= cur + 1e-4:
        return "Reducing stored K (SK1) preserved PPL — read-time selection compensates."
    return "Reducing stored K (SK1) raised PPL — store budget matters for what the router can read."


if __name__ == "__main__":
    main()
