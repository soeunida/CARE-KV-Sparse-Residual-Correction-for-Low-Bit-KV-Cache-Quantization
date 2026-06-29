"""tools/eval_phase11b_longseq_adaptive_carekv.py — Phase 11B.

Long-sequence adaptive CARE-KV study. Goal: understand why CARE-KV's quality gain
over BaseQuant_INT3 diminishes at longer sequence lengths, and test lightweight
adaptive policies that might recover it. Quality (PPL) only — NO decode-speed /
fused-attention claim, NO TurboQuant+CAREKV combination.

Three experiment groups, all at fixed INT3 base, fp16 residual, vectorized=1, uniform
per-layer budget. Reference uniform default = SK2SV4 (store 2,4 / read 2,2):

  Group A  budget scaling      : SK0SV8, SK2SV4, SK4SV4, SK2SV8, SK4SV8
                                 (does spending more residual budget recover the gain?)
  Group B  selector upper bound: current / random / oracle_residual_magnitude /
                                 oracle_reconstruction_error  (at SK2SV4)
                                 (how far is the current scorer from an oracle selector?)
  Group C  position-aware alloc: recent_token / prefix_sink / sink_plus_recent /
                                 middle_drop  (at SK2SV4, same total budget)
                                 (does biasing residuals by token position help long seq?)

Group B uses the gated CAREKV_SELECTOR_VARIANT (store-side K selection) + the existing
router CAREKV_BASELINE_SCORE ablation hook (read-side V selection). Group C uses the
gated CAREKV_POSITION_POLICY + CAREKV_SEQ_LEN (read-side V re-weight). All gated knobs
default to no-op → byte-identical to the paper-best path when unset.

Reference rows fp16 / BaseQuant_INT3 / TurboQuant_INT3_standalone are computed per
(model, seq_len) for deltas. Memory columns are ANALYTIC estimates (labeled as such).

Run (NS=4 smoke first; bump to 8 for the main rows):
  CUDA_VISIBLE_DEVICES=5 python tools/eval_phase11b_longseq_adaptive_carekv.py \
    --out_dir results/phase11b_longseq_adaptive_carekv \
    --num_samples 4 --seq_lens 512 1024 \
    --models mistralai/Mistral-7B-v0.3 01-ai/Yi-6B --resume
"""
import os, sys, csv, time, math, argparse, importlib.util
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
_z = importlib.util.spec_from_file_location("ezoo", "tools/eval_adaptive_model_zoo.py")
ezoo = importlib.util.module_from_spec(_z); _z.loader.exec_module(ezoo)
eac, eap = ezoo.eac, ezoo.eap
from CARE_KV.care_kv.baselines.common import eval_ppl_wikitext
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter

# budget tag -> (store_k, store_v, read_k, read_v, kind). SK2SV4 keeps the canonical
# read(2,2) so V-read selection (2 of 4 stored blocks) has selection pressure; the
# other tags read what they store (capped internally: K groups<=4, V blocks<=4).
BUDGETS = {
    "SK0SV8": (0, 8, 0, 8, "v"),
    "SK2SV4": (2, 4, 2, 2, "both"),   # canonical paper-best default
    "SK4SV4": (4, 4, 4, 4, "both"),
    "SK2SV8": (2, 8, 2, 8, "both"),
    "SK4SV8": (4, 8, 4, 8, "both"),
}
# selector variant -> router baseline_score (read-side V selection mechanism)
SELECTOR_BASELINE = {
    "current": "carekv",
    "random": "random",
    "oracle_residual_magnitude": "magnitude_only",
    "oracle_reconstruction_error": "oracle_proxy",
}

# CARE-KV cell specs (in run order). A_SK2SV4 doubles as the uniform reference and the
# Group-B "current" selector baseline.
CELLS = [
    # ── Group A: budget scaling ──
    {"method": "CAREKV_A_SK0SV8", "group": "A_budget", "budget": "SK0SV8"},
    {"method": "CAREKV_A_SK2SV4", "group": "A_budget", "budget": "SK2SV4"},   # = uniform ref / B current
    {"method": "CAREKV_A_SK4SV4", "group": "A_budget", "budget": "SK4SV4"},
    {"method": "CAREKV_A_SK2SV8", "group": "A_budget", "budget": "SK2SV8"},
    {"method": "CAREKV_A_SK4SV8", "group": "A_budget", "budget": "SK4SV8"},
    # ── Group B: selector upper bound (at SK2SV4) ──
    {"method": "CAREKV_B_random", "group": "B_selector", "budget": "SK2SV4", "selector": "random"},
    {"method": "CAREKV_B_oracle_residual_magnitude", "group": "B_selector", "budget": "SK2SV4",
     "selector": "oracle_residual_magnitude"},
    {"method": "CAREKV_B_oracle_reconstruction_error", "group": "B_selector", "budget": "SK2SV4",
     "selector": "oracle_reconstruction_error"},
    # ── Group C: position-aware allocation (at SK2SV4) ──
    {"method": "CAREKV_C_recent_token", "group": "C_position", "budget": "SK2SV4", "position": "recent_token"},
    {"method": "CAREKV_C_prefix_sink", "group": "C_position", "budget": "SK2SV4", "position": "prefix_sink"},
    {"method": "CAREKV_C_sink_plus_recent", "group": "C_position", "budget": "SK2SV4", "position": "sink_plus_recent"},
    {"method": "CAREKV_C_middle_drop", "group": "C_position", "budget": "SK2SV4", "position": "middle_drop"},
]
UNIFORM_REF = "CAREKV_A_SK2SV4"          # delta_vs_uniform_SK2SV4 base
ORACLE_METHODS = {"CAREKV_B_oracle_residual_magnitude", "CAREKV_B_oracle_reconstruction_error"}

COLS = ["model_id", "seq_len", "num_samples", "method", "group", "effective_budget",
        "selector_variant", "position_policy", "ppl", "delta_vs_fp16",
        "delta_vs_basequant_int3", "delta_vs_turboquant_int3", "delta_vs_uniform_SK2SV4",
        "oracle_gap", "residual_value_MB", "residual_metadata_MB", "total_carekv_MB",
        "extra_MB_over_base", "K_reads", "V_reads", "peak_gpu_mem_MB", "wall_time_s",
        "status", "failure_reason", "oom", "paper_usable", "notes"]

# Phase 11B gated env vars to clear between cells so settings never leak.
_PH11B_ENV = ["CAREKV_SELECTOR_VARIANT", "CAREKV_BASELINE_SCORE", "CAREKV_POSITION_POLICY",
              "CAREKV_SEQ_LEN"]


def _clear_11b_env():
    for k in _PH11B_ENV:
        os.environ.pop(k, None)


def _preset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def _peak():
    return round(torch.cuda.max_memory_allocated() / 1e6, 2) if torch.cuda.is_available() else ""


def carekv_mem_estimate(cfg, S, store_k, store_v, bits=3, sketch_dim=8):
    """Analytic per-context (single sample of length S) memory estimate, in MB.
    Returns (residual_value_MB, residual_metadata_MB, total_carekv_MB, extra_MB_over_base).
    int4 residual payload = 0.5 B/value; int8 scale = 1 B; fp16 sketch/errnorm = 2 B.
    Caps: K channel groups <= head_dim/32, V token blocks <= page_size/4 = 4."""
    L = getattr(cfg, "num_hidden_layers", 32)
    Hkv = getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 32)
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    page, kcg, vtb = 16, 32, 4
    npages = math.ceil(S / page)
    sk = min(store_k, max(1, D // kcg)) if store_k > 0 else 0
    sv = min(store_v, page // vtb) if store_v > 0 else 0
    per = L * Hkv * npages
    kval = sk * page * kcg * 0.5            # int4 K residual payload
    vval = sv * vtb * D * 0.5               # int4 V residual payload
    res_val = per * (kval + vval)
    kmeta = sk * (sketch_dim * 2 + 1)       # K sketch (fp16) + int8 scale
    vmeta = sv * (2 + 1)                    # V error-norm (fp16) + int8 scale
    res_meta = per * (kmeta + vmeta) + per * (sk + sv) * 4   # + slot-map ints
    base = 2 * L * Hkv * S * D * (bits / 8.0)               # packed INT base K+V
    base += 2 * L * Hkv * npages * 1                        # per-page int8 master scale
    mb = 1e6
    return (round(res_val / mb, 3), round(res_meta / mb, 3),
            round((base + res_val + res_meta) / mb, 3), round((res_val + res_meta) / mb, 3))


def run_turbo(mid, bits, sl, ns):
    ad = TurboQuantStyleAdapter(bits_k=bits, bits_v=bits, qjl_m=0, use_qjl=True)
    try:
        m = ad.setup_model(mid); tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id or 0
        ppl, _ = eval_ppl_wikitext(m, tok, sl, ns); ad.teardown(); del m; torch.cuda.empty_cache()
        return ppl, None
    except torch.cuda.OutOfMemoryError as e:
        try: ad.teardown()
        except Exception: pass
        torch.cuda.empty_cache(); return None, ("oom", str(e)[:60])
    except Exception as e:
        try: ad.teardown()
        except Exception: pass
        torch.cuda.empty_cache(); return None, ("err", f"{type(e).__name__}: {str(e)[:60]}")


def row(mid, sl, ns, **kw):
    r = {c: "" for c in COLS}
    r.update(model_id=mid, seq_len=sl, num_samples=ns, oom="no")
    r.update({k: v for k, v in kw.items() if k in COLS})
    return r


def run_model_seq(mid, sl, A):
    """All reference + CARE-KV cells for one (model, seq_len). Returns list of rows."""
    eap.SL, eap.NS = sl, A.num_samples
    os.environ["CAREKV_MAX_PAGES"] = str(max(40, math.ceil(sl / 16) + 8))
    mtype = AutoConfig.from_pretrained(mid).model_type
    p = ezoo.port_params(mtype, mid)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(p.get("dtype"), torch.float16)
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    model_cfg = AutoConfig.from_pretrained(mid)

    def load():
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, device_map="cuda")
        m.config.use_cache = False; m.eval(); return m

    rows = []
    print(f"[11b {mid} SL{sl}] start (mtype={mtype}, NS={A.num_samples})", flush=True)

    # ── fp16 reference ──
    _clear_11b_env(); _preset(); t0 = time.perf_counter()
    m = load(); cfg = m.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    p["gs"], p["kcg"], p["packed"] = ezoo.group_for_head_dim(hd)
    ref = eval_ppl_wikitext(m, tok, sl, A.num_samples)[0]; del m; torch.cuda.empty_cache()
    rows.append(row(mid, sl, A.num_samples, method="fp16", group="reference", effective_budget="-",
                    ppl=round(ref, 4), delta_vs_fp16=0.0, peak_gpu_mem_MB=_peak(),
                    wall_time_s=round(time.perf_counter() - t0, 1), status="real",
                    paper_usable="yes", notes="full-precision reference"))
    print(f"[11b {mid} SL{sl}] fp16={ref:.4f}", flush=True)

    # ── Gate A + BaseQuant INT3 + Gate B base ──
    _clear_11b_env()
    g, *_ = eac.run_cell(load, tok, p, "fp", "uniform", A.bit, 0, 0, 0, 0, "both")
    gateA = math.isfinite(g) and abs(g - ref) < max(0.02 * ref, 0.02)
    _preset()
    b3, _kr, _vr, rt_b3, _e = eac.run_cell(load, tok, p, "base_quant", "uniform", A.bit, 0, 0, 0, 0, "both")
    silent = math.isfinite(b3) and abs(b3 - ref) < 1e-3
    collapse = math.isfinite(b3) and b3 > 5 * ref
    rows.append(row(mid, sl, A.num_samples, method="BaseQuant_INT3", group="reference",
                    effective_budget="SK0SV0", ppl=round(b3, 4) if math.isfinite(b3) else "nan",
                    delta_vs_fp16=round(b3 - ref, 4) if math.isfinite(b3) else "",
                    K_reads=0, V_reads=0, peak_gpu_mem_MB=_peak(), wall_time_s=rt_b3,
                    status="blocked_architecture_port" if silent else ("unstable_outlier_collapse" if collapse else "real"),
                    paper_usable="no" if (silent or collapse) else "yes",
                    failure_reason="silent HF fallback (arch not ported)" if silent else ("INT3 collapse" if collapse else ""),
                    notes="compression-only baseline"))
    base_usable = math.isfinite(b3) and not silent and not collapse
    print(f"[11b {mid} SL{sl}] base3={b3:.4f} gateA={gateA} silent={silent} collapse={collapse}", flush=True)

    # ── TurboQuant INT3 standalone (for delta_vs_turboquant) ──
    _clear_11b_env(); _preset(); t0 = time.perf_counter()
    tq, terr = run_turbo(mid, A.bit, sl, A.num_samples); pk_tq = _peak(); wt = round(time.perf_counter() - t0, 1)
    tq_ok = terr is None and tq is not None and math.isfinite(tq)
    coll_tq = tq_ok and tq > 5 * ref
    rows.append(row(mid, sl, A.num_samples, method="TurboQuant_INT3_standalone", group="reference",
                    effective_budget="-", ppl=round(tq, 4) if tq_ok else "nan",
                    delta_vs_fp16=round(tq - ref, 4) if tq_ok else "",
                    delta_vs_basequant_int3=round(tq - b3, 4) if (tq_ok and math.isfinite(b3)) else "",
                    peak_gpu_mem_MB=pk_tq, wall_time_s=wt, oom="yes" if (terr and terr[0] == "oom") else "no",
                    status="unstable_outlier_collapse" if coll_tq else ("real" if tq_ok else
                           ("blocked_oom" if (terr and terr[0] == "oom") else "blocked_architecture_port")),
                    paper_usable="no" if (coll_tq or not tq_ok) else "yes",
                    failure_reason="" if tq_ok else (terr[1] if terr else "non-finite"),
                    notes="TurboQuant-style standalone (QJL)"))
    tqv = tq if tq_ok else None
    print(f"[11b {mid} SL{sl}] TurboQuant_INT3={tq if tq_ok else terr}", flush=True)

    # ── CARE-KV cells ──
    os.environ["CAREKV_VECTORIZED_RESIDUAL"] = "1"
    os.environ["CAREKV_VECTORIZE_VDOM_ONLY"] = "1"
    os.environ["CAREKV_DEBUG_STATS"] = "1"
    for k in ("CAREKV_RESIDUAL_DTYPE_PPL_SMOKE", "CAREKV_RESIDUAL_DTYPE", "CAREKV_RESIDUAL_SCALE_MODE"):
        os.environ.pop(k, None)
    uniform_ppl = {}   # filled when A_SK2SV4 runs

    cells = CELLS if not A.cells else [c for c in CELLS if c["method"] in set(A.cells)]
    for spec in cells:
        sk, sv, rk, rv, kind = BUDGETS[spec["budget"]]
        selector = spec.get("selector", "current")
        position = spec.get("position", "none")
        _clear_11b_env()
        os.environ["CAREKV_SEQ_LEN"] = str(sl)
        if selector != "current":
            os.environ["CAREKV_SELECTOR_VARIANT"] = selector
            os.environ["CAREKV_BASELINE_SCORE"] = SELECTOR_BASELINE[selector]
        if position != "none":
            os.environ["CAREKV_POSITION_POLICY"] = position
        try:
            # Gate B (READ0 == BaseQuant) only for the canonical default cell.
            gateB = True
            if spec["method"] == UNIFORM_REF:
                r0, *_ = eac.run_cell(load, tok, p, "carekv_stored", "uniform", A.bit, sk, sv, 0, 0, kind)
                gateB = math.isfinite(b3) and math.isfinite(r0) and abs(r0 - b3) < 1e-2
            _preset(); t0 = time.perf_counter()
            ck, ckr, ckv, rt_ck, e_ck = eac.run_cell(load, tok, p, "carekv_stored", "uniform", A.bit,
                                                     sk, sv, rk, rv, kind)
            pk = _peak()
            fired = (ckr + ckv) > 0
            ck_ok = gateA and gateB and base_usable and math.isfinite(ck) and fired
            rv_mb, rmeta_mb, tot_mb, extra_mb = carekv_mem_estimate(model_cfg, sl, sk, sv, bits=A.bit)
            rows.append(row(
                mid, sl, A.num_samples, method=spec["method"], group=spec["group"],
                effective_budget=spec["budget"], selector_variant=selector, position_policy=position,
                ppl=round(ck, 4) if math.isfinite(ck) else "",
                delta_vs_fp16=round(ck - ref, 4) if math.isfinite(ck) else "",
                delta_vs_basequant_int3=round(ck - b3, 4) if (math.isfinite(ck) and math.isfinite(b3)) else "",
                delta_vs_turboquant_int3=round(ck - tqv, 4) if (math.isfinite(ck) and tqv is not None) else "",
                residual_value_MB=rv_mb, residual_metadata_MB=rmeta_mb, total_carekv_MB=tot_mb,
                extra_MB_over_base=extra_mb, K_reads=ckr, V_reads=ckv, peak_gpu_mem_MB=pk,
                wall_time_s=rt_ck,
                status="real" if ck_ok else ("blocked_architecture_port" if (not gateA or not base_usable)
                                             else "blocked_router_silent" if not fired else "blocked"),
                paper_usable="yes" if ck_ok else "no",
                failure_reason="" if ck_ok else ("gate A / base unusable" if (not gateA or not base_usable)
                                                 else "gate B fail" if not gateB else
                                                 "router never fired (0 reads)" if not fired else "non-finite"),
                notes=f"{spec['group']} budget={spec['budget']} selector={selector} position={position}; "
                      f"gateA={gateA} gateB={gateB}"))
            if spec["method"] == UNIFORM_REF and math.isfinite(ck):
                uniform_ppl[(mid, sl)] = ck
            print(f"[11b {mid} SL{sl}] {spec['method']}={ck if math.isfinite(ck) else 'nan'} "
                  f"reads(K={ckr},V={ckv}) ok={ck_ok}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rows.append(row(mid, sl, A.num_samples, method=spec["method"], group=spec["group"],
                            effective_budget=spec["budget"], selector_variant=selector,
                            position_policy=position, status="blocked_oom", oom="yes",
                            paper_usable="no", failure_reason=str(e)[:80], notes="OOM"))
            print(f"[11b {mid} SL{sl}] {spec['method']} OOM", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(row(mid, sl, A.num_samples, method=spec["method"], group=spec["group"],
                            effective_budget=spec["budget"], selector_variant=selector,
                            position_policy=position, status="blocked", paper_usable="no",
                            failure_reason=f"{type(e).__name__}: {str(e)[:80]}", notes="error"))
            print(f"[11b {mid} SL{sl}] {spec['method']} ERROR {e}", flush=True)
    _clear_11b_env()

    # ── post: delta_vs_uniform_SK2SV4 + oracle_gap ──
    u = uniform_ppl.get((mid, sl))
    for r in rows:
        try:
            x = float(r["ppl"])
        except Exception:
            continue
        if r.get("group", "").startswith(("A_", "B_", "C_")) and u is not None and math.isfinite(u):
            r["delta_vs_uniform_SK2SV4"] = round(x - u, 4)
        if r["method"] in ORACLE_METHODS and u is not None and math.isfinite(u):
            r["oracle_gap"] = round(u - x, 4)   # current_selector_ppl - oracle_ppl (>0 = oracle better)
    return rows


def write_outputs(rows, A):
    all_csv = os.path.join(A.out_dir, "phase11b_all_rows.csv")
    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow(r)
    # group-specific summaries
    def _flt(r, g): return r.get("group") == g and r.get("ppl") not in ("", "nan")
    bcols_A = ["model_id", "seq_len", "method", "effective_budget", "ppl", "delta_vs_fp16",
               "delta_vs_basequant_int3", "delta_vs_uniform_SK2SV4", "extra_MB_over_base",
               "K_reads", "V_reads", "paper_usable"]
    with open(os.path.join(A.out_dir, "phase11b_budget_scaling.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bcols_A, extrasaction="ignore"); w.writeheader()
        for r in rows:
            if _flt(r, "A_budget"):
                w.writerow(r)
    bcols_B = ["model_id", "seq_len", "method", "selector_variant", "ppl", "delta_vs_uniform_SK2SV4",
               "oracle_gap", "K_reads", "V_reads", "paper_usable"]
    with open(os.path.join(A.out_dir, "phase11b_selector_oracle_gap.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bcols_B, extrasaction="ignore"); w.writeheader()
        for r in rows:
            if r.get("method") == UNIFORM_REF or _flt(r, "B_selector"):
                w.writerow(r)
    bcols_C = ["model_id", "seq_len", "method", "position_policy", "ppl", "delta_vs_uniform_SK2SV4",
               "delta_vs_basequant_int3", "K_reads", "V_reads", "paper_usable"]
    with open(os.path.join(A.out_dir, "phase11b_position_policy.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bcols_C, extrasaction="ignore"); w.writeheader()
        for r in rows:
            if r.get("method") == UNIFORM_REF or _flt(r, "C_position"):
                w.writerow(r)

    rep = ["# Phase 11B — long-sequence adaptive CARE-KV\n",
           f"> {len(A.models)} models x {A.seq_lens} x {len(CELLS)} CARE-KV cells + 3 references, "
           f"NS={A.num_samples}, INT3 base, fp16 residual, vectorized=1, uniform budget. "
           "Quality (PPL) only — no decode-speed / fused-attention claim; no TurboQuant+CAREKV.\n",
           "\n## Question this phase answers\n",
           "Why does CARE-KV's gain over BaseQuant_INT3 shrink at longer seq_len, and can a "
           "lightweight adaptive policy (more budget / better selector / position bias) recover it?\n",
           "\n## Groups\n",
           "- **A budget_scaling** — SK0SV8/SK2SV4/SK4SV4/SK2SV8/SK4SV8. Does more residual budget help?\n",
           "- **B selector_oracle_gap** — current vs random vs oracle (residual-magnitude / "
           "reconstruction-error) at SK2SV4. `oracle_gap = current_ppl - oracle_ppl` (>0 ⇒ headroom).\n",
           "- **C position_policy** — recent_token / prefix_sink / sink_plus_recent / middle_drop at "
           "SK2SV4 (same total budget, position-biased V read).\n",
           f"\nRows: {len(rows)}. CSVs: phase11b_all_rows / phase11b_budget_scaling / "
           "phase11b_selector_oracle_gap / phase11b_position_policy.\n",
           "\n_Memory columns are analytic per-context estimates (int4 payload 0.5 B/val, "
           "int8 scale, fp16 sketch/err-norm), not measured allocations._\n"]
    open(os.path.join(A.out_dir, "PHASE11B_REPORT.md"), "w").write("\n".join(rep) + "\n")
    print(f"[11b] wrote {len(rows)} rows + 3 group CSVs + report to {A.out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--models", nargs="+",
                    default=["mistralai/Mistral-7B-v0.3", "01-ai/Yi-6B"])
    ap.add_argument("--bit", type=int, default=3)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="subset of CARE-KV cell method names to run (default: all 12). "
                         "References fp16/BaseQuant_INT3/TurboQuant always run.")
    ap.add_argument("--resume", action="store_true")
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    all_csv = os.path.join(A.out_dir, "phase11b_all_rows.csv")

    existing, done = [], set()
    if A.resume and os.path.exists(all_csv):
        existing = list(csv.DictReader(open(all_csv)))
        done = {(r["model_id"], r["seq_len"]) for r in existing}
        print(f"[11b] resume: {len(done)} (model,seq_len) blocks already done", flush=True)

    rows = list(existing)
    for mid in A.models:
        for sl in A.seq_lens:
            if (mid, str(sl)) in done:
                print(f"[11b] skip {mid} SL{sl} (resume)", flush=True); continue
            try:
                rows += run_model_seq(mid, sl, A)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rows.append(row(mid, sl, A.num_samples, method="ALL", group="reference",
                                status="blocked_oom", oom="yes", paper_usable="no",
                                failure_reason=str(e)[:80], notes="OOM"))
                print(f"[11b {mid} SL{sl}] block OOM", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                rows.append(row(mid, sl, A.num_samples, method="ALL", group="reference",
                                status="blocked", paper_usable="no",
                                failure_reason=f"{type(e).__name__}: {str(e)[:80]}", notes="error"))
                print(f"[11b {mid} SL{sl}] block ERROR {e}", flush=True)
            write_outputs(rows, A)   # incremental
    write_outputs(rows, A)
    print("[11b] done.", flush=True)


if __name__ == "__main__":
    main()
