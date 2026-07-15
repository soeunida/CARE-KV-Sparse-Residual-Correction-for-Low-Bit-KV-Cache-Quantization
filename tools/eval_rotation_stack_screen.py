"""tools/eval_rotation_stack_screen.py

Screening: does Hadamard pre-RoPE rotation + read-all-V STACK on top of the
current best `combined_exact` (KSCORE_LIVE + exact K correction) to widen the
thin margin over TurboQuant on the hard diffuse-error models?

This is the experiment the pre-2026-07-15 rotation docs never ran: all prior
rotation screening used the OLD uniform-base / linear-K config. Here every
CARE-KV arm carries KSCORE_LIVE=1 + K_CORRECTION_MODE=exact (today's paper-best
levers), so we isolate what rotation + read-breadth ADD on top of it.

Arms (all CARE-KV arms: combined + exact, correction_impl=vectorized):
  turbo        TurboQuant INT3 reference (rotation + QJL baseline)
  bar          uniform base, post-RoPE, rv=2  (== carekv_combined_exact, current best)
  uni_rv4      uniform base, post-RoPE, rv=4  (read-all-V; read-breadth WITHOUT rotation)
  rot_rv4      rotatekv_style base, pre-RoPE, rv=4  (Hadamard pre-RoPE + read-all-V — hypothesis)

Same run_one harness / WT-2 windowing as tools/eval_exact_kcorr.py, so PPLs are
directly comparable to the §10 exact_kcorr tables.

Run (GPU 6 is free):
  CUDA_VISIBLE_DEVICES=6 python tools/eval_rotation_stack_screen.py \
    --out_csv results/rotation_stack/deepseek7b_ns8.csv --num-samples 8 \
    --models deepseek-ai/deepseek-llm-7b-base
"""
import os, sys, csv, time, math, argparse
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/care_kv_clean/tools")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from CARE_KV.care_kv.baselines import FP16Adapter, BaseQuantAdapter, CAREKVAdapter
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from eval_base_quantizer_expansion import run_one


def maxp_for(sl):
    return max(16, math.ceil(sl / 16) + 8)


def carekv(maxp, base_quantizer="uniform", k_store_mode="post_rope", rv=2):
    # sv=4 stored V slots; rv=4 == read-all-V, rv=2 == paper read budget.
    return CAREKVAdapter(mode="fixed", bits=3, base_quantizer=base_quantizer,
                         k_store_mode=k_store_mode, bits_k=3, bits_v=3,
                         sk=2, sv=4, rk=2, rv=rv, max_pages=maxp,
                         correction_impl="vectorized")


# arm -> (adapter, kscore_live, k_correction_mode)
def arm_spec(arm, maxp):
    if arm == "fp16":     return FP16Adapter(), False, None
    if arm == "turbo":    return TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=0, use_qjl=True), False, None
    if arm == "bar":      return carekv(maxp, "uniform", "post_rope", rv=2), True, "exact"
    if arm == "uni_rv4":  return carekv(maxp, "uniform", "post_rope", rv=4), True, "exact"
    if arm == "rot_rv4":  return carekv(maxp, "rotatekv_style", "pre_rope", rv=4), True, "exact"
    raise ValueError(arm)


ALL_ARMS = ["fp16", "turbo", "bar", "uni_rv4", "rot_rv4"]
COLS = ["model_id", "seq_len", "num_samples", "arm", "ppl", "delta_vs_fp16",
        "delta_vs_turbo", "delta_vs_bar", "k_reads", "v_reads", "runtime_s", "status"]


def fv(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=ALL_ARMS)
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)
    rows, done = [], set()
    if os.path.exists(A.out_csv):
        rows = list(csv.DictReader(open(A.out_csv)))
        done = {(r["model_id"], r["seq_len"], r["arm"]) for r in rows}

    def flush():
        with open(A.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
            for r in rows: w.writerow(r)

    print(f"[rotstack] models={A.models} SL{A.seq_lens} N{A.num_samples} arms={A.arms}", flush=True)
    for mid in A.models:
        for sl in A.seq_lens:
            maxp = maxp_for(sl)
            for arm in A.arms:
                if (mid, str(sl), arm) in done:
                    print(f"[rotstack] skip {mid} SL{sl} {arm}", flush=True); continue
                os.environ.pop("CAREKV_KSCORE_LIVE", None)
                os.environ.pop("CAREKV_K_CORRECTION_MODE", None)
                ad, kscore, kmode = arm_spec(arm, maxp)
                if kscore: os.environ["CAREKV_KSCORE_LIVE"] = "1"
                if kmode:  os.environ["CAREKV_K_CORRECTION_MODE"] = kmode
                t0 = time.perf_counter()
                try:
                    r = run_one(ad, mid, "wikitext", sl, A.num_samples)
                    ppl = float(r.ppl) if (r.ppl and math.isfinite(float(r.ppl))) else float("nan")
                    rec = dict(model_id=mid, seq_len=sl, num_samples=A.num_samples, arm=arm,
                               ppl=round(ppl, 4) if math.isfinite(ppl) else "nan",
                               k_reads=r.k_reads, v_reads=r.v_reads,
                               runtime_s=round(time.perf_counter() - t0, 1),
                               status="real" if math.isfinite(ppl) else "nonfinite")
                    print(f"[rotstack] {mid.split('/')[-1]:22s} SL{sl} {arm:9s} PPL={rec['ppl']}  "
                          f"K={r.k_reads} V={r.v_reads}  ({rec['runtime_s']}s)", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    rec = dict(model_id=mid, seq_len=sl, num_samples=A.num_samples, arm=arm,
                               status=f"error:{type(e).__name__}")
                os.environ.pop("CAREKV_KSCORE_LIVE", None)
                os.environ.pop("CAREKV_K_CORRECTION_MODE", None)
                rows.append(rec); flush()

            def g(a):
                for r in rows:
                    if r["model_id"] == mid and str(r["seq_len"]) == str(sl) and r["arm"] == a:
                        return fv(r.get("ppl"))
                return None
            fp, tq, bar = g("fp16"), g("turbo"), g("bar")
            for r in rows:
                if r["model_id"] == mid and str(r["seq_len"]) == str(sl) and fv(r.get("ppl")) is not None:
                    p = fv(r["ppl"])
                    r["delta_vs_fp16"] = round(p - fp, 4) if fp else ""
                    r["delta_vs_turbo"] = round(p - tq, 4) if tq else ""
                    r["delta_vs_bar"] = round(p - bar, 4) if bar else ""
            flush()

    print("\n=== rotation + read-all-V stacked on combined_exact ===", flush=True)
    hdr = f"{'model':22s} {'SL':>5} {'turbo':>8} {'bar':>8} {'uni_rv4':>8} {'rot_rv4':>8} {'rot-bar':>8} {'rot-turbo':>9}"
    print(hdr); print("-" * len(hdr), flush=True)
    for mid in A.models:
        for sl in A.seq_lens:
            def g(a):
                for r in rows:
                    if r["model_id"] == mid and str(r["seq_len"]) == str(sl) and r["arm"] == a:
                        return fv(r.get("ppl"))
                return None
            tq, bar, uni, rot = g("turbo"), g("bar"), g("uni_rv4"), g("rot_rv4")
            if None in (tq, bar, rot): continue
            print(f"{mid.split('/')[-1]:22s} {sl:>5} {tq:>8.4f} {bar:>8.4f} "
                  f"{(uni if uni else float('nan')):>8.4f} {rot:>8.4f} "
                  f"{rot-bar:>+8.4f} {rot-tq:>+9.4f}", flush=True)
    print(f"\n[rotstack] done -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
