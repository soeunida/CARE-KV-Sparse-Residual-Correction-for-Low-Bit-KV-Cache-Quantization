"""tools/eval_kstab_screening.py — K-correction stabilization screening.

Tests whether the gated K-stabilization knobs prevent the outlier-driven CARE-KV
K-correction blow-up (Phase 11C: opt-1.3b collapsed to 231, Δbase=+206) AND narrow
the TurboQuant gap on outlier-heavy models (Yi / DeepSeek).

Knobs (both default 0 = OFF, read in the vectorized correction path):
  CAREKV_K_QDOTR_CLAMP_PCT : clamp |q·R_K| to its pXX percentile (tame Jacobian spikes)
  CAREKV_K_NORM_GUARD_PCT  : skip K slots whose residual norm exceeds the pXX percentile

For each model × K-stab setting at SK2SV4 (SL512/NS4): PPL, Δvs Base, Δvs Turbo, dK/dV
correction norms, collapse flag, K/V reads. K correction scale is RESTORED to 0.1
(eval_arch_ports.set_env otherwise forces it to 0.0 → would make the knobs meaningless).

Run:
  CUDA_VISIBLE_DEVICES=5 python tools/eval_kstab_screening.py \
    --out_dir results/kstab_screening --seq_len 512 --num_samples 4
"""
import os, sys, csv, time, math, argparse, importlib.util
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
_z = importlib.util.spec_from_file_location("ezoo", "tools/eval_adaptive_model_zoo.py")
ezoo = importlib.util.module_from_spec(_z); _z.loader.exec_module(ezoo)
eac, eap = ezoo.eac, ezoo.eap

# Restore K-correction scale after set_env (which hardcodes 0.0 = V-only).
K_CORRECTION_SCALE = os.environ.get("KSTAB_K_CORRECTION_SCALE", "0.1")
_orig_set_env = eap.set_env
def _set_env_kfix(*a, **k):
    _orig_set_env(*a, **k)
    os.environ["CAREKV_K_CORRECTION_SCALE"] = K_CORRECTION_SCALE
eap.set_env = _set_env_kfix

from CARE_KV.care_kv.baselines.common import eval_ppl_wikitext
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from CARE_KV.care_kv.layer import get_debug_stats

# K-stab settings: (label, env overrides)
KSTAB = [
    ("baseline",          {}),
    ("clamp_p99",         {"CAREKV_K_QDOTR_CLAMP_PCT": "99"}),
    ("clamp_p95",         {"CAREKV_K_QDOTR_CLAMP_PCT": "95"}),
    ("nguard_p99",        {"CAREKV_K_NORM_GUARD_PCT": "99"}),
    ("nguard_p95",        {"CAREKV_K_NORM_GUARD_PCT": "95"}),
    ("clamp99_nguard99",  {"CAREKV_K_QDOTR_CLAMP_PCT": "99", "CAREKV_K_NORM_GUARD_PCT": "99"}),
]
_KSTAB_ENV = ["CAREKV_K_QDOTR_CLAMP_PCT", "CAREKV_K_NORM_GUARD_PCT"]

COLS = ["model_id", "family", "seq_len", "num_samples", "kstab", "ppl", "delta_vs_fp16",
        "delta_vs_basequant_int3", "delta_vs_turboquant_int3", "delta_vs_baseline_carekv",
        "correction_delta_norm_K", "correction_delta_norm_V", "K_reads", "V_reads",
        "collapse", "status", "paper_usable", "notes"]


def fam_of(mid):
    s = mid.lower()
    if "opt" in s: return "OPT"
    if "deepseek" in s: return "DeepSeek"
    if "yi-" in s: return "Yi"
    if "open_llama" in s: return "OpenLLaMA"
    return "LLaMA"


def finite_collapse(ppl, fp):
    return (math.isfinite(ppl) and math.isfinite(fp) and ppl > max(100.0, 5.0 * fp))


def run_turbo(mid, sl, ns):
    ad = TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=0, use_qjl=True)
    try:
        m = ad.setup_model(mid); tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
        ppl, _ = eval_ppl_wikitext(m, tok, sl, ns); ad.teardown(); del m; torch.cuda.empty_cache()
        return ppl
    except Exception:
        try: ad.teardown()
        except Exception: pass
        torch.cuda.empty_cache(); return None


def row(**kw):
    r = {c: "" for c in COLS}; r.update(kw); return r


def run_model(mid, sl, ns, bit):
    fam = fam_of(mid)
    eap.SL, eap.NS = sl, ns
    os.environ["CAREKV_MAX_PAGES"] = str(max(40, math.ceil(sl / 16) + 8))
    cfg0 = AutoConfig.from_pretrained(mid)
    p = ezoo.port_params(cfg0.model_type, mid)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(p.get("dtype"), torch.float16)
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
    def load():
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, device_map="cuda")
        m.config.use_cache = False; m.eval(); return m
    m = load(); hd = getattr(m.config, "head_dim", None) or m.config.hidden_size // m.config.num_attention_heads
    p["gs"], p["kcg"], p["packed"] = ezoo.group_for_head_dim(hd)
    ref = eval_ppl_wikitext(m, tok, sl, ns)[0]; del m; torch.cuda.empty_cache()
    rows = [row(model_id=mid, family=fam, seq_len=sl, num_samples=ns, kstab="fp16",
                ppl=round(ref, 4), status="real", paper_usable="yes", notes="reference")]
    print(f"[kstab {mid}] fp16={ref:.4f}", flush=True)

    b3, *_ = eac.run_cell(load, tok, p, "base_quant", "uniform", bit, 0, 0, 0, 0, "both")
    base_collapse = finite_collapse(b3, ref)
    rows.append(row(model_id=mid, family=fam, seq_len=sl, num_samples=ns, kstab="BaseQuant_INT3",
                    ppl=round(b3, 4), delta_vs_fp16=round(b3 - ref, 4), status="real",
                    paper_usable="no" if base_collapse else "yes",
                    notes="base collapse" if base_collapse else "compression baseline"))
    tq = run_turbo(mid, sl, ns)
    tq_ok = tq is not None and math.isfinite(tq) and not finite_collapse(tq, ref)
    rows.append(row(model_id=mid, family=fam, seq_len=sl, num_samples=ns, kstab="TurboQuant_INT3",
                    ppl=round(tq, 4) if tq is not None else "n/a",
                    delta_vs_fp16=round(tq - ref, 4) if tq_ok else "", status="real" if tq_ok else "blocked",
                    paper_usable="yes" if tq_ok else "no", notes="QJL standalone"))
    print(f"[kstab {mid}] base3={b3:.4f} turbo={tq} base_collapse={base_collapse}", flush=True)

    # CARE-KV cells across K-stab settings
    os.environ["CAREKV_VECTORIZED_RESIDUAL"] = "1"; os.environ["CAREKV_VECTORIZE_VDOM_ONLY"] = "1"
    os.environ["CAREKV_DEBUG_STATS"] = "1"
    baseline_ppl = None
    for label, env in KSTAB:
        for k in _KSTAB_ENV: os.environ.pop(k, None)
        for k, v in env.items(): os.environ[k] = v
        try:
            t0 = time.perf_counter()
            ck, ckr, ckv, rt, err = eac.run_cell(load, tok, p, "carekv_stored", "uniform", bit, 2, 4, 2, 2, "both")
            st = get_debug_stats()
        except Exception as e:
            rows.append(row(model_id=mid, family=fam, seq_len=sl, num_samples=ns, kstab=label,
                            status="blocked", paper_usable="no", notes=f"{type(e).__name__}: {str(e)[:60]}"))
            print(f"[kstab {mid}] {label} ERROR {e}", flush=True); continue
        dK = float(st.get("delta_k_norm_sum", 0.0)); dV = float(st.get("delta_v_norm_sum", 0.0))
        coll = finite_collapse(ck, ref)
        if label == "baseline" and math.isfinite(ck): baseline_ppl = ck
        usable = (not coll and not base_collapse and math.isfinite(ck) and (ckr + ckv) > 0)
        rows.append(row(model_id=mid, family=fam, seq_len=sl, num_samples=ns, kstab=label,
                        ppl=round(ck, 4) if math.isfinite(ck) else "",
                        delta_vs_fp16=round(ck - ref, 4) if math.isfinite(ck) else "",
                        delta_vs_basequant_int3=round(ck - b3, 4) if (math.isfinite(ck) and math.isfinite(b3)) else "",
                        delta_vs_turboquant_int3=round(ck - tq, 4) if (math.isfinite(ck) and tq_ok) else "",
                        delta_vs_baseline_carekv=round(ck - baseline_ppl, 4) if (baseline_ppl and math.isfinite(ck)) else "",
                        correction_delta_norm_K=round(dK, 3), correction_delta_norm_V=round(dV, 3),
                        K_reads=ckr, V_reads=ckv, collapse="yes" if coll else "no",
                        status="unstable_outlier_collapse" if coll else "real",
                        paper_usable="yes" if usable else "no",
                        notes=f"{label}; t={round(time.perf_counter()-t0)}s"))
        print(f"[kstab {mid}] {label} ppl={ck if math.isfinite(ck) else 'nan'} dK={dK:.1f} "
              f"collapse={coll} reads(K={ckr},V={ckv})", flush=True)
    for k in _KSTAB_ENV: os.environ.pop(k, None)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/kstab_screening")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--bit", type=int, default=3)
    ap.add_argument("--models", nargs="+",
                    default=["facebook/opt-1.3b", "deepseek-ai/deepseek-llm-7b-base", "01-ai/Yi-6B"])
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    all_csv = os.path.join(A.out_dir, "kstab_all_rows.csv")
    rows = []
    print(f"[kstab] models={A.models} SL{A.seq_len} NS{A.num_samples} K_scale={K_CORRECTION_SCALE}", flush=True)
    for mid in A.models:
        try:
            rows += run_model(mid, A.seq_len, A.num_samples, A.bit)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rows.append(row(model_id=mid, family=fam_of(mid), kstab="ALL", status="blocked_oom",
                            paper_usable="no", notes=str(e)[:60]))
            print(f"[kstab {mid}] OOM", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(row(model_id=mid, family=fam_of(mid), kstab="ALL", status="blocked",
                            paper_usable="no", notes=f"{type(e).__name__}: {str(e)[:60]}"))
        with open(all_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
            for r in rows: w.writerow(r)
    print(f"[kstab] done. {len(rows)} rows -> {all_csv}", flush=True)


if __name__ == "__main__":
    main()
