"""tools/eval_kscore_ablation.py — Phase-3 query-aware K-score SMOKE ablation.

Flag-gated query-aware K-side residual scoring (CAREKV_KSCORE_LIVE):
  ΔK = K_fp − K_quant ; ΔS = Q·ΔKᵀ/√d ; sensitivity = A·(1−A) ;
  KScore = ‖ΔS·sensitivity‖·‖V‖ ; CombinedScore = λ_k·norm(KScore)+λ_v·norm(VScore).

Variants:
  vscore_only      — current V-score-only path (CAREKV_KSCORE_LIVE=0, V-dominant).
  kscore_only      — query-aware K-score only (KSCORE_LIVE=1, K-dominant).
  combined_kvscore — K+V combined with λ_k / λ_v (KSCORE_LIVE=1).

SMOKE ONLY: Mistral-7B-v0.3, seq_len 128/256, num_samples 4. Not the full grid.
Default check: vscore_only uses KSCORE_LIVE=0 and must equal the existing V-only
CARE-KV result. Failures/OOMs preserved. Not committed.
"""
import os, sys, csv, math, time, argparse, importlib.util
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
_spec = importlib.util.spec_from_file_location("eap", "tools/eval_arch_ports.py")
eap = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eap)
from CARE_KV.care_kv.layer import reset_debug_stats, get_debug_stats

D = "results/kscore_ablation/smoke"
MID = "mistralai/Mistral-7B-v0.3"
P = dict(family="mistral", dtype="bf16", gs=32, kcg=32, packed=1)

# variant -> (kscore_live, kind, read_k, read_v, effective_budget)
VARIANTS = {
    "vscore_only":      dict(live="0", kind="v",    rk=0, rv=2, budget="SK0SV4"),
    "kscore_only":      dict(live="1", kind="k",    rk=2, rv=0, budget="SK2SV0"),
    "combined_kvscore": dict(live="1", kind="both", rk=2, rv=2, budget="SK2SV4"),
}
COLS = ["model_id", "seq_len", "num_samples", "variant", "kscore_live", "kind",
        "effective_budget", "lambda_k", "lambda_v", "ppl", "delta_vs_base",
        "k_reads", "v_reads", "selected_correction", "kscore_changed_decision",
        "status", "failure_reason", "runtime_s", "notes"]


def append(rows, path):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        if new: w.writeheader()
        for r in rows: w.writerow({c: r.get(c, "") for c in COLS})


def run_cell(load, tok, prefill, bq, bits, sk, sv, rk, rv, kind, live, lam_k, lam_v):
    m = load()
    eap.set_env(P, prefill, bq, bits, sk, sv, rk, rv, kind)
    os.environ["CAREKV_KSCORE_LIVE"] = live
    os.environ["CAREKV_KSCORE_LAMBDA_K"] = str(lam_k)
    os.environ["CAREKV_KSCORE_LAMBDA_V"] = str(lam_v)
    os.environ["CAREKV_RESIDUAL_SCORE"] = "raw_error"   # V-side proxy (K-score is the variable here)
    eap.patch(m, P, bits)
    from CARE_KV.care_kv.llama_patch import reset_all_caches
    reset_all_caches(m); reset_debug_stats()
    t0 = time.time()
    try:
        pv = eap.ppl(m, tok); err = None
    except torch.cuda.OutOfMemoryError as e:
        pv = float("nan"); err = ("oom", str(e)[:60])
    except Exception as e:
        pv = float("nan"); err = ("err", f"{type(e).__name__}: {str(e)[:60]}")
    rt = round(time.time() - t0, 1)
    st = get_debug_stats(); kr, vr = int(st.get("k_slots_read", 0)), int(st.get("v_slots_read", 0))
    del m; torch.cuda.empty_cache()
    return pv, kr, vr, rt, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--seq-lens", nargs="*", type=int, default=[128, 256])
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--lambda-k", type=float, default=1.0)
    ap.add_argument("--lambda-v", type=float, default=1.0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.makedirs(D, exist_ok=True)
    raw = f"{D}/kscore_ablation_all_rows.csv"
    print(f"=== K-score SMOKE ablation GPU {a.gpu}  Mistral-7B  SL={a.seq_lens} NS={a.num_samples} "
          f"λk={a.lambda_k} λv={a.lambda_v} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
    dtype = torch.bfloat16

    def loader(SL, NS):
        def load():
            m = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=dtype, device_map="cuda")
            m.config.use_cache = False; m.eval(); return m
        return load

    all_rows = []
    for SL in a.seq_lens:
        eap.SL, eap.NS = SL, a.num_samples
        load = loader(SL, a.num_samples)
        # fp16 + BaseQuant_INT3 base for context
        m = load(); ref = eap.ppl(m, tok); del m; torch.cuda.empty_cache()
        b3, *_ = run_cell(load, tok, "base_quant", "uniform", 3, 0, 0, 0, 0, "both", "0", 1, 1)
        print(f"[SL{SL}] fp16={ref:.4f} base_INT3={b3:.4f}", flush=True)
        all_rows.append(dict(model_id=MID, seq_len=SL, num_samples=a.num_samples, variant="fp16",
                             kscore_live="-", kind="-", effective_budget="-", ppl=round(ref, 4),
                             status="real", notes="reference"))
        all_rows.append(dict(model_id=MID, seq_len=SL, num_samples=a.num_samples, variant="BaseQuant_INT3",
                             kscore_live="-", kind="-", effective_budget="SK0SV0", ppl=round(b3, 4),
                             status="real", notes="INT3 base"))
        results = {}
        for vname, v in VARIANTS.items():
            pv, kr, vr, rt, err = run_cell(load, tok, "carekv_stored", "uniform", 3, 2, 4,
                                           v["rk"], v["rv"], v["kind"], v["live"], a.lambda_k, a.lambda_v)
            status = "real" if (err is None and math.isfinite(pv)) else (
                "blocked_oom" if (err and err[0] == "oom") else "blocked")
            fr = (err[1] if err else "")
            results[vname] = pv if math.isfinite(pv) else None
            all_rows.append(dict(model_id=MID, seq_len=SL, num_samples=a.num_samples, variant=vname,
                                 kscore_live=v["live"], kind=v["kind"], effective_budget=v["budget"],
                                 lambda_k=a.lambda_k if v["live"] == "1" else "",
                                 lambda_v=a.lambda_v if v["live"] == "1" else "",
                                 ppl=round(pv, 4) if math.isfinite(pv) else "nan",
                                 delta_vs_base=round(pv - b3, 4) if (math.isfinite(pv) and math.isfinite(b3)) else "",
                                 k_reads=kr, v_reads=vr, selected_correction=vname,
                                 status=status, failure_reason=fr, runtime_s=rt,
                                 notes="SMOKE ablation (not the full grid)"))
            print(f"[SL{SL}] {vname:18} ppl={all_rows[-1]['ppl']} K{kr}/V{vr} [{status}]", flush=True)
        # decision: best variant; whether K-score changed it vs vscore_only
        valid = {k: v for k, v in results.items() if v is not None}
        if valid:
            best = min(valid, key=valid.get)
            changed = (best != "vscore_only")
            for r in all_rows:
                if r["seq_len"] == SL and r["variant"] in VARIANTS:
                    r["kscore_changed_decision"] = ("yes" if changed else "no")
            print(f"[SL{SL}] best variant={best}  K-score changed decision vs vscore_only: {'YES' if changed else 'no'}", flush=True)
        append([r for r in all_rows if r["seq_len"] == SL], raw)

    # ---- summary CSV ----
    scols = ["seq_len", "fp16", "base_int3", "vscore_only", "kscore_only", "combined_kvscore",
             "best_variant", "kscore_changed_decision", "combined_beats_vscore"]
    summ = []
    for SL in a.seq_lens:
        g = {r["variant"]: r for r in all_rows if r["seq_len"] == SL}
        def pp(v): return g.get(v, {}).get("ppl", "")
        vs = g.get("vscore_only", {}).get("ppl"); cb = g.get("combined_kvscore", {}).get("ppl")
        def fnum(x):
            try: return float(x)
            except: return None
        beats = ""
        if fnum(vs) is not None and fnum(cb) is not None:
            beats = "yes" if fnum(cb) < fnum(vs) else "no"
        valid = {v: fnum(g.get(v, {}).get("ppl")) for v in VARIANTS if fnum(g.get(v, {}).get("ppl")) is not None}
        best = min(valid, key=valid.get) if valid else ""
        summ.append(dict(seq_len=SL, fp16=pp("fp16"), base_int3=pp("BaseQuant_INT3"),
                         vscore_only=pp("vscore_only"), kscore_only=pp("kscore_only"),
                         combined_kvscore=pp("combined_kvscore"), best_variant=best,
                         kscore_changed_decision=("yes" if best and best != "vscore_only" else "no"),
                         combined_beats_vscore=beats))
    with open(f"{D}/kscore_ablation_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scols); w.writeheader()
        for r in summ: w.writerow(r)

    # ---- report ----
    o = ["# K-score ablation (Phase 3) — SMOKE\n",
         "> **Smoke ablation, NOT the final full grid.** Model Mistral-7B-v0.3, seq_len "
         f"{a.seq_lens}, num_samples {a.num_samples}. Flag-gated query-aware K-side residual score "
         "(`CAREKV_KSCORE_LIVE`); with `CAREKV_KSCORE_LIVE=0` the path is byte-identical to the current "
         "V-score-only CARE-KV. Variants: vscore_only / kscore_only / combined_kvscore "
         f"(λ_k={a.lambda_k}, λ_v={a.lambda_v}). Failures/OOMs preserved.\n",
         "## PPL by variant and seq_len\n",
         "| SL | fp16 | base INT3 | vscore_only | kscore_only | combined_kvscore | best | K-score changed decision | combined beats vscore |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in summ:
        o.append(f"| {r['seq_len']} | {r['fp16']} | {r['base_int3']} | {r['vscore_only']} | {r['kscore_only']} "
                 f"| {r['combined_kvscore']} | **{r['best_variant']}** | {r['kscore_changed_decision']} | {r['combined_beats_vscore']} |")
    o.append("\n## Effective budgets\n- vscore_only → SK0SV4 (V-dominant)\n- kscore_only → SK2SV0 (K-dominant)\n"
             "- combined_kvscore → SK2SV4 (K+V)\n")
    fails = [r for r in all_rows if r.get("status", "").startswith("blocked")]
    o.append(f"## Failures / OOM (preserved): {len(fails)}\n")
    for r in fails:
        o.append(f"- {r['variant']} SL{r['seq_len']}: {r['status']} ({r.get('failure_reason','')})")
    o.append("\n## Notes\n- This is a SMOKE ablation; do not read it as the final full K-score grid.\n"
             "- `CAREKV_KSCORE_LIVE=0` (vscore_only) is the unchanged default V-score-only path.\n")
    open(f"{D}/KSCORE_ABLATION_REPORT.md", "w").write("\n".join(o) + "\n")
    print("KSCORE_ABLATION_DONE", flush=True)


if __name__ == "__main__":
    main()
