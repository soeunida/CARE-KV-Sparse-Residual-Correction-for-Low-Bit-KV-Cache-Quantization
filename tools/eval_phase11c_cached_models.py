"""tools/eval_phase11c_cached_models.py — Phase 11C.

CARE-KV characterization on ALREADY-CACHED models only (no downloads). Offline-
enforced (HF_HUB_OFFLINE=1 / P11C_ALLOW_DL=0). For each cached model at SL512/NS4:
references (fp16, BaseQuant_INT3, TurboQuant_INT3 if it loads) + CARE-KV current
SK2SV4, with full correction instrumentation. combined_kvscore is run only for
Mistral-family models (none are currently cached → not run here).

Per-row instrumentation (from layer.get_debug_stats()):
  K_reads, V_reads, correction_delta_norm_K (Σ‖k_corr ΔO‖), correction_delta_norm_V
  (Σ‖V ΔO‖), k_correction_active (K delta norm>0 AND K slots read>0), correction_type
  (K+V / V-dominant / skip).

HARD STOP (campaign halts, writes partial CSV) on:
  - ENOSPC          : free disk < P11C_MIN_FREE_GB (default 3 GB)
  - OOM             : CUDA OOM on any cell
  - inactive K corr : a Mistral-family kind=both cell that passed Gate A but has
                      k_correction_active=False (genuine K-correction regression)
  - invalid K/V corr: non-finite correction delta norm
  - READ=0 violation: Gate B (carekv READ0 != BaseQuant) fails on a model whose
                      router fired

Arch-not-ported (Gate A fail OR router never fires: K_reads+V_reads==0) is a clean
per-model SKIP (status=blocked_architecture_port), NOT a campaign stop.

Run:
  CUDA_VISIBLE_DEVICES=5 P11C_ALLOW_DL=0 python tools/eval_phase11c_cached_models.py \
    --out_dir results/phase11c_cached_models --seq_len 512 --num_samples 4
"""
import os, sys, csv, time, math, argparse, shutil, importlib.util
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# offline: never download (P11C_ALLOW_DL=0 default).
if os.environ.get("P11C_ALLOW_DL", "0") != "1":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
_z = importlib.util.spec_from_file_location("ezoo", "tools/eval_adaptive_model_zoo.py")
ezoo = importlib.util.module_from_spec(_z); _z.loader.exec_module(ezoo)
eac, eap = ezoo.eac, ezoo.eap

# eap.set_env (eval_arch_ports.py) hardcodes CAREKV_K_CORRECTION_SCALE="0.0" (V-only).
# Without this wrapper, K correction is OFF for every cell → dK==0, k_correction_active
# always False, everything mislabeled "V-dominant". Restore the layer.py default 0.1
# after set_env so "current SK2SV4" is the faithful K+V path and the K instrumentation
# is meaningful (mirrors Phase 11A's _set_env_kfix).
K_CORRECTION_SCALE = os.environ.get("P11C_K_CORRECTION_SCALE", "0.1")
_orig_set_env = eap.set_env
def _set_env_kfix(*a, **k):
    _orig_set_env(*a, **k)
    os.environ["CAREKV_K_CORRECTION_SCALE"] = K_CORRECTION_SCALE
eap.set_env = _set_env_kfix

from CARE_KV.care_kv.baselines.common import eval_ppl_wikitext
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from CARE_KV.care_kv.layer import get_debug_stats

# priority order; uncached entries are skipped at runtime (cached-only rule).
PRIORITY = [
    "mistralai/Mistral-7B-Instruct-v0.3",      # Mistral-family (likely uncached → skip)
    "HuggingFaceH4/zephyr-7b-beta",            # Mistral-family
    "teknium/OpenHermes-2.5-Mistral-7B",       # Mistral-family
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "openlm-research/open_llama_3b_v2",
    "Qwen/Qwen2.5-7B",
    "openlm-research/open_llama_7b_v2",
    "deepseek-ai/deepseek-llm-7b-base",
]
MISTRAL_FAMILY = ("mistral", "zephyr", "hermes", "Mistral", "Zephyr", "Hermes")


def family_of(mid):
    s = mid.lower()
    if any(t.lower() in s for t in ("mistral", "zephyr", "hermes")):
        return "Mistral-family"
    if "opt" in s:
        return "OPT"
    if "open_llama" in s or "openllama" in s:
        return "OpenLLaMA"
    if "qwen" in s:
        return "Qwen"
    if "deepseek" in s:
        return "DeepSeek(LLaMA)"
    return "LLaMA"


def is_mistral_family(mid):
    return family_of(mid) == "Mistral-family"


COLS = ["model_id", "family", "model_type", "seq_len", "num_samples", "method", "ppl",
        "delta_vs_basequant_int3", "delta_vs_turboquant_int3", "K_reads", "V_reads",
        "correction_delta_norm_K", "correction_delta_norm_V", "k_correction_active",
        "correction_type", "peak_gpu_mem_MB", "wall_time_s", "status", "failure_reason",
        "paper_usable", "notes"]


def is_cached(mid):
    """True iff the model snapshot exists locally (no network)."""
    try:
        from huggingface_hub import scan_cache_dir
        return mid in {r.repo_id for r in scan_cache_dir().repos
                       if any(rev.snapshots for rev in [r])}
    except Exception:
        # fallback: check default cache dir for the repo folder w/ weights
        d = os.path.expanduser("~/.cache/huggingface/hub/models--" + mid.replace("/", "--"))
        import glob
        return bool(glob.glob(d + "/snapshots/*/*.safetensors") or glob.glob(d + "/snapshots/*/*.bin"))


def free_gb(path="."):
    return shutil.disk_usage(path).free / 1e9


def finite_collapse(ppl, fp):
    """Finite-but-collapsed PPL guard: ppl & fp16 finite AND ppl > max(100, 5*fp16).
    Catches K-correction blow-ups (opt-1.3b 231 vs fp16 21) and INT3 base collapses
    (Qwen) that the bare isfinite() check would pass as paper_usable."""
    return (math.isfinite(ppl) and math.isfinite(fp) and ppl > max(100.0, 5.0 * fp))


def _preset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def _peak():
    return round(torch.cuda.max_memory_allocated() / 1e6, 2) if torch.cuda.is_available() else ""


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


def row(mid, fam, mt, sl, ns, **kw):
    r = {c: "" for c in COLS}
    r.update(model_id=mid, family=fam, model_type=mt, seq_len=sl, num_samples=ns)
    r.update({k: v for k, v in kw.items() if k in COLS})
    return r


class HardStop(Exception):
    def __init__(self, reason): super().__init__(reason); self.reason = reason


def run_model(mid, sl, ns, bit, min_free_gb):
    """All cells for one model. Returns list of rows. Raises HardStop on stop-conditions."""
    fam = family_of(mid)
    if free_gb() < min_free_gb:
        raise HardStop(f"ENOSPC: free {free_gb():.1f}GB < {min_free_gb}GB before {mid}")
    cfg0 = AutoConfig.from_pretrained(mid)
    mt = cfg0.model_type
    eap.SL, eap.NS = sl, ns
    os.environ["CAREKV_MAX_PAGES"] = str(max(40, math.ceil(sl / 16) + 8))
    p = ezoo.port_params(mt, mid)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(p.get("dtype"), torch.float16)
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    def load():
        m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dtype, device_map="cuda")
        m.config.use_cache = False; m.eval(); return m

    rows = []
    print(f"[11c {mid}] start fam={fam} type={mt}", flush=True)

    # ── fp16 reference ──
    _preset(); t0 = time.perf_counter()
    try:
        m = load(); cfg = m.config
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        p["gs"], p["kcg"], p["packed"] = ezoo.group_for_head_dim(hd)
        ref = eval_ppl_wikitext(m, tok, sl, ns)[0]; del m; torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache(); raise HardStop(f"OOM loading/eval fp16 {mid}: {str(e)[:60]}")
    rows.append(row(mid, fam, mt, sl, ns, method="fp16", ppl=round(ref, 4),
                    peak_gpu_mem_MB=_peak(), wall_time_s=round(time.perf_counter() - t0, 1),
                    status="real", paper_usable="yes", correction_type="-", notes="reference"))
    print(f"[11c {mid}] fp16={ref:.4f}", flush=True)

    # ── Gate A (fp-mode == HF) + BaseQuant_INT3 ──
    try:
        g, *_ = eac.run_cell(load, tok, p, "fp", "uniform", bit, 0, 0, 0, 0, "both")
        gateA = math.isfinite(g) and abs(g - ref) < max(0.02 * ref, 0.02)
        _preset()
        b3, _kr, _vr, rt_b3, _e = eac.run_cell(load, tok, p, "base_quant", "uniform", bit, 0, 0, 0, 0, "both")
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache(); raise HardStop(f"OOM gateA/base {mid}: {str(e)[:60]}")
    silent = math.isfinite(b3) and abs(b3 - ref) < 1e-3
    collapse = finite_collapse(b3, ref)   # finite-but-collapsed INT3 base (e.g. Qwen)
    rows.append(row(mid, fam, mt, sl, ns, method="BaseQuant_INT3",
                    ppl=round(b3, 4) if math.isfinite(b3) else "nan",
                    K_reads=0, V_reads=0, peak_gpu_mem_MB=_peak(), wall_time_s=rt_b3,
                    correction_type="-",
                    status="blocked_architecture_port" if silent else ("unstable_outlier_collapse" if collapse else "real"),
                    paper_usable="no" if (silent or collapse) else "yes",
                    failure_reason="silent HF fallback (arch not ported)" if silent else ("INT3 collapse" if collapse else ""),
                    notes="compression-only baseline"))
    base_usable = math.isfinite(b3) and not silent and not collapse
    print(f"[11c {mid}] base3={b3:.4f} gateA={gateA} silent={silent} collapse={collapse}", flush=True)

    # ── TurboQuant_INT3 (best-effort; may not support this arch) ──
    _preset(); t0 = time.perf_counter()
    tq, terr = run_turbo(mid, bit, sl, ns); pk_tq = _peak(); wt = round(time.perf_counter() - t0, 1)
    tq_ok = terr is None and tq is not None and math.isfinite(tq)
    tq_collapse = tq_ok and finite_collapse(tq, ref)
    # base-collapsed setting OR turbo's own finite collapse → not a usable positive
    tq_usable = tq_ok and not tq_collapse and base_usable
    tq_reason = ("" if tq_usable else
                 ("turbo_finite_collapse" if tq_collapse else
                  "base_collapsed_setting" if (tq_ok and not base_usable) else
                  (terr[1] if terr else "non-finite")))
    rows.append(row(mid, fam, mt, sl, ns, method="TurboQuant_INT3_standalone",
                    ppl=round(tq, 4) if tq_ok else "n/a",
                    delta_vs_basequant_int3=round(tq - b3, 4) if (tq_ok and math.isfinite(b3)) else "",
                    peak_gpu_mem_MB=pk_tq, wall_time_s=wt, correction_type="-",
                    status=("unstable_outlier_collapse" if tq_collapse else "real") if tq_ok
                           else ("blocked_oom" if (terr and terr[0] == "oom") else "blocked_architecture_port"),
                    paper_usable="yes" if tq_usable else "no",
                    failure_reason=tq_reason,
                    notes="TurboQuant-style standalone (QJL)"))
    if terr and terr[0] == "oom":
        raise HardStop(f"OOM TurboQuant {mid}")
    print(f"[11c {mid}] turbo={tq if tq_ok else terr}", flush=True)

    # ── CARE-KV current SK2SV4 (+ Gate B + correction instrumentation) ──
    os.environ["CAREKV_VECTORIZED_RESIDUAL"] = "1"; os.environ["CAREKV_VECTORIZE_VDOM_ONLY"] = "1"
    os.environ["CAREKV_DEBUG_STATS"] = "1"
    for k in ("CAREKV_RESIDUAL_DTYPE_PPL_SMOKE", "CAREKV_RESIDUAL_DTYPE", "CAREKV_RESIDUAL_SCALE_MODE",
              "CAREKV_SELECTOR_VARIANT", "CAREKV_POSITION_POLICY", "CAREKV_BASELINE_SCORE"):
        os.environ.pop(k, None)
    try:
        # Gate B: READ0 must equal BaseQuant
        r0, r0k, r0v, _rt, _e = eac.run_cell(load, tok, p, "carekv_stored", "uniform", bit, 2, 4, 0, 0, "both")
        gateB = math.isfinite(b3) and math.isfinite(r0) and abs(r0 - b3) < 1e-2
        _preset(); t0 = time.perf_counter()
        ck, ckr, ckv, rt_ck, e_ck = eac.run_cell(load, tok, p, "carekv_stored", "uniform", bit, 2, 4, 2, 2, "both")
        st = get_debug_stats()
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache(); raise HardStop(f"OOM CARE-KV {mid}: {str(e)[:60]}")
    dK = float(st.get("delta_k_norm_sum", 0.0)); dV = float(st.get("delta_v_norm_sum", 0.0))
    fired = (ckr + ckv) > 0
    k_active = (dK > 1e-9) and (st.get("k_slots_read", 0) > 0)
    finite_corr = math.isfinite(dK) and math.isfinite(dV)

    # correction type
    if not (gateA and base_usable and fired):
        corr_type = "skip"
    elif k_active and dV > 0 and dK >= 0.2 * dV:
        corr_type = "K+V"
    elif dV > 0 or ckv > 0:
        corr_type = "V-dominant"
    elif k_active:
        corr_type = "K-only"
    else:
        corr_type = "none"

    ck_collapse = finite_collapse(ck, ref)          # finite-but-collapsed CARE-KV (e.g. opt-1.3b K-blowup)
    ck_ok = (gateA and gateB and base_usable and math.isfinite(ck) and fired
             and finite_corr and not ck_collapse)
    rows.append(row(mid, fam, mt, sl, ns, method="CAREKV_current_SK2SV4",
                    ppl=round(ck, 4) if math.isfinite(ck) else "",
                    delta_vs_basequant_int3=round(ck - b3, 4) if (math.isfinite(ck) and math.isfinite(b3)) else "",
                    delta_vs_turboquant_int3=round(ck - tq, 4) if (math.isfinite(ck) and tq_ok) else "",
                    K_reads=ckr, V_reads=ckv,
                    correction_delta_norm_K=round(dK, 4), correction_delta_norm_V=round(dV, 4),
                    k_correction_active=k_active, correction_type=corr_type,
                    peak_gpu_mem_MB=_peak(), wall_time_s=rt_ck,
                    status=("unstable_outlier_collapse" if (ck_collapse and base_usable) else
                            "real" if ck_ok else ("blocked_architecture_port" if not (gateA and fired)
                                                  else "blocked")),
                    paper_usable="yes" if ck_ok else "no",
                    failure_reason="" if ck_ok else (
                        "arch not ported / router never fired" if not (gateA and fired)
                        else "gate B (READ0!=base) fail" if not gateB
                        else "base_collapsed_setting" if not base_usable
                        else "non-finite correction" if not finite_corr
                        else ("carekv_finite_collapse (k_correction_blowup)" if ck_collapse
                              and dK > dV else "carekv_finite_collapse") if ck_collapse
                        else "not usable"),
                    notes=f"current SK2SV4; gateA={gateA} gateB={gateB} corr_type={corr_type}"
                          + (" COLLAPSE" if ck_collapse else "")))
    print(f"[11c {mid}] CAREKV={ck if math.isfinite(ck) else 'nan'} reads(K={ckr},V={ckv}) "
          f"dK={dK:.3f} dV={dV:.3f} k_active={k_active} type={corr_type}", flush=True)

    # ── HARD STOP conditions on the CARE-KV cell ──
    if not finite_corr:
        raise HardStop(f"invalid (non-finite) K/V correction norm on {mid} (dK={dK}, dV={dV})")
    if fired and base_usable and not gateB:
        raise HardStop(f"READ=0 invariant violation on {mid}: READ0 ppl {r0:.4f} != base {b3:.4f}")
    if is_mistral_family(mid) and gateA and fired and not k_active:
        raise HardStop(f"inactive K correction on Mistral-family {mid} (dK={dK})")

    # ── combined_kvscore: Mistral-family only (none cached → typically skipped) ──
    if is_mistral_family(mid) and ck_ok:
        os.environ["CAREKV_KSCORE_LIVE"] = "1"
        try:
            _preset(); t0 = time.perf_counter()
            cb, cbr, cbv, rt_cb, _e = eac.run_cell(load, tok, p, "carekv_stored", "uniform", bit, 2, 4, 2, 2, "both")
            st2 = get_debug_stats()
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache(); os.environ.pop("CAREKV_KSCORE_LIVE", None)
            raise HardStop(f"OOM combined_kvscore {mid}")
        os.environ.pop("CAREKV_KSCORE_LIVE", None)
        dK2 = float(st2.get("delta_k_norm_sum", 0.0)); dV2 = float(st2.get("delta_v_norm_sum", 0.0))
        rows.append(row(mid, fam, mt, sl, ns, method="CAREKV_combined_kvscore",
                        ppl=round(cb, 4) if math.isfinite(cb) else "",
                        delta_vs_basequant_int3=round(cb - b3, 4) if (math.isfinite(cb) and math.isfinite(b3)) else "",
                        delta_vs_turboquant_int3=round(cb - tq, 4) if (math.isfinite(cb) and tq_ok) else "",
                        K_reads=cbr, V_reads=cbv, correction_delta_norm_K=round(dK2, 4),
                        correction_delta_norm_V=round(dV2, 4),
                        k_correction_active=(dK2 > 1e-9 and st2.get("k_slots_read", 0) > 0),
                        correction_type="K+V", peak_gpu_mem_MB=_peak(), wall_time_s=rt_cb,
                        status="real", paper_usable="yes",
                        notes="combined_kvscore (CAREKV_KSCORE_LIVE=1)"))
        print(f"[11c {mid}] combined_kvscore={cb if math.isfinite(cb) else 'nan'}", flush=True)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results/phase11c_cached_models")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--bit", type=int, default=3)
    ap.add_argument("--min_free_gb", type=float, default=float(os.environ.get("P11C_MIN_FREE_GB", "3")))
    ap.add_argument("--resume", action="store_true")
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    all_csv = os.path.join(A.out_dir, "phase11c_all_rows.csv")

    existing, done = [], set()
    if A.resume and os.path.exists(all_csv):
        existing = list(csv.DictReader(open(all_csv)))
        done = {r["model_id"] for r in existing}

    # cached-only selection (priority order)
    runnable, skipped = [], []
    for mid in PRIORITY:
        (runnable if is_cached(mid) else skipped).append(mid)
    print(f"[11c] cached/runnable ({len(runnable)}): {runnable}", flush=True)
    print(f"[11c] uncached/skipped ({len(skipped)}): {skipped}", flush=True)
    print(f"[11c] free disk {free_gb():.1f}GB (guard {A.min_free_gb}GB); offline={os.environ.get('HF_HUB_OFFLINE')}", flush=True)

    rows = list(existing)
    # record skipped (uncached) models
    for mid in skipped:
        if mid in done:
            continue
        rows.append(row(mid, family_of(mid), "?", A.seq_len, A.num_samples, method="ALL",
                        status="skipped_not_cached", paper_usable="no",
                        failure_reason="not in local HF cache (download disabled)", correction_type="skip",
                        notes="cached-only rule"))

    stop_reason = None
    for mid in runnable:
        if mid in done:
            print(f"[11c] skip {mid} (resume)", flush=True); continue
        try:
            rows += run_model(mid, A.seq_len, A.num_samples, A.bit, A.min_free_gb)
        except HardStop as hs:
            rows.append(row(mid, family_of(mid), "?", A.seq_len, A.num_samples, method="ALL",
                            status="HARD_STOP", paper_usable="no", failure_reason=hs.reason,
                            correction_type="skip", notes="campaign halted"))
            stop_reason = hs.reason
            print(f"[11c] *** HARD STOP at {mid}: {hs.reason} ***", flush=True)
            break
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(row(mid, family_of(mid), "?", A.seq_len, A.num_samples, method="ALL",
                            status="blocked", paper_usable="no",
                            failure_reason=f"{type(e).__name__}: {str(e)[:80]}", correction_type="skip",
                            notes="error (continue)"))
        # incremental write
        with open(all_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
            for r in rows:
                w.writerow(r)

    with open(all_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow(r)
    rep = ["# Phase 11C — CARE-KV on cached models (no downloads)\n",
           f"> SL{A.seq_len}/NS{A.num_samples}, INT3. Cached-only; offline enforced. "
           f"Runnable: {runnable}. Skipped (uncached): {skipped}.\n",
           f"\nHARD STOP: {stop_reason or 'none — completed all runnable models'}.\n",
           f"\nRows: {len(rows)}. See phase11c_all_rows.csv.\n"]
    open(os.path.join(A.out_dir, "PHASE11C_REPORT.md"), "w").write("\n".join(rep) + "\n")
    print(f"[11c] done. rows={len(rows)} stop={stop_reason}", flush=True)


if __name__ == "__main__":
    main()
