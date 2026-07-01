"""tools/eval_combined_vs_turbo.py — combined_kvscore vs TurboQuant confirmation (B).

The one lead where CARE-KV might beat TurboQuant: the combined_kvscore selector
(CAREKV_KSCORE_LIVE=1, query-aware K+V score) on Mistral. Prior small-NS runs had
combined beating both the current selector and Turbo on Mistral, but the Turbo reference
was noisy. This confirms at higher NS in the SAME run_one harness (so Turbo/CARE-KV share
windowing) whether combined_kvscore actually beats TurboQuant.

Arms: fp16 / BaseQuant_INT3 / TurboQuant_INT3 / CARE-KV current SK2SV4 / CARE-KV
combined_kvscore SK2SV4 (KSCORE_LIVE=1). Uniform-packed base, vectorized correction.

Run:
  CUDA_VISIBLE_DEVICES=2 python tools/eval_combined_vs_turbo.py \
    --out_csv results/combined_vs_turbo/cvt.csv --num-samples 32 --seq-lens 512 1024 \
    --models mistralai/Mistral-7B-v0.3
"""
import os, sys, csv, time, math, argparse
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/CARE_KV/care_kv/tools")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from CARE_KV.care_kv.baselines import FP16Adapter, BaseQuantAdapter, CAREKVAdapter
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from eval_base_quantizer_expansion import run_one


def maxp_for(sl):
    return max(16, math.ceil(sl / 16) + 8)


def carekv(maxp):
    return CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform", k_store_mode="post_rope",
                         bits_k=3, bits_v=3, sk=2, sv=4, rk=2, rv=2, max_pages=maxp,
                         correction_impl="vectorized")


# arm -> (adapter factory, kscore_live)
def arm_spec(arm, maxp):
    if arm == "fp16":             return FP16Adapter(), False
    if arm == "base_int3":        return BaseQuantAdapter(bits=3), False
    if arm == "turbo_int3":       return TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=0, use_qjl=True), False
    if arm == "carekv_current":   return carekv(maxp), False
    if arm == "carekv_combined":  return carekv(maxp), True      # KSCORE_LIVE=1
    raise ValueError(arm)


ARMS = ["fp16", "base_int3", "turbo_int3", "carekv_current", "carekv_combined"]
COLS = ["model_id", "seq_len", "num_samples", "arm", "ppl", "delta_vs_fp16",
        "delta_vs_turbo", "delta_vs_current", "k_reads", "v_reads", "runtime_s", "status"]


def fv(x):
    try: return float(x)
    except Exception: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--models", nargs="+", default=["mistralai/Mistral-7B-v0.3"])
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

    print(f"[cvt] models={A.models} SL{A.seq_lens} N{A.num_samples}", flush=True)
    for mid in A.models:
        for sl in A.seq_lens:
            maxp = maxp_for(sl)
            for arm in ARMS:
                if (mid, str(sl), arm) in done:
                    print(f"[cvt] skip {mid} SL{sl} {arm}", flush=True); continue
                os.environ.pop("CAREKV_KSCORE_LIVE", None)
                ad, kscore = arm_spec(arm, maxp)
                if kscore:
                    os.environ["CAREKV_KSCORE_LIVE"] = "1"
                t0 = time.perf_counter()
                try:
                    r = run_one(ad, mid, "wikitext", sl, A.num_samples)
                    ppl = float(r.ppl) if (r.ppl and math.isfinite(float(r.ppl))) else float("nan")
                    rec = dict(model_id=mid, seq_len=sl, num_samples=A.num_samples, arm=arm,
                               ppl=round(ppl, 4) if math.isfinite(ppl) else "nan",
                               k_reads=r.k_reads, v_reads=r.v_reads,
                               runtime_s=round(time.perf_counter() - t0, 1),
                               status="real" if math.isfinite(ppl) else "nonfinite")
                    print(f"[cvt] {mid.split('/')[-1]:20s} SL{sl} {arm:16s} PPL={rec['ppl']}  "
                          f"K={r.k_reads} V={r.v_reads}  ({rec['runtime_s']}s)", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    rec = dict(model_id=mid, seq_len=sl, num_samples=A.num_samples, arm=arm,
                               status=f"error:{type(e).__name__}")
                os.environ.pop("CAREKV_KSCORE_LIVE", None)
                rows.append(rec); flush()
            # deltas for this (model, sl)
            def g(a):
                for r in rows:
                    if r["model_id"] == mid and r["seq_len"] == str(sl) and r["arm"] == a:
                        return fv(r.get("ppl"))
                return None
            fp, tq, cur = g("fp16"), g("turbo_int3"), g("carekv_current")
            for r in rows:
                if r["model_id"] == mid and r["seq_len"] == str(sl) and fv(r.get("ppl")) is not None:
                    p = fv(r["ppl"])
                    r["delta_vs_fp16"] = round(p - fp, 4) if fp else ""
                    r["delta_vs_turbo"] = round(p - tq, 4) if tq else ""
                    r["delta_vs_current"] = round(p - cur, 4) if cur else ""
            flush()
    # verdict
    print("\n=== combined_kvscore vs Turbo ===", flush=True)
    for mid in A.models:
        for sl in A.seq_lens:
            def g(a):
                for r in rows:
                    if r["model_id"] == mid and r["seq_len"] == str(sl) and r["arm"] == a:
                        return fv(r.get("ppl"))
                return None
            cur, cmb, tq = g("carekv_current"), g("carekv_combined"), g("turbo_int3")
            if None not in (cur, cmb, tq):
                print(f"[cvt] {mid.split('/')[-1]} SL{sl}: current={cur:.4f} combined={cmb:.4f} "
                      f"turbo={tq:.4f} | combined-turbo={cmb-tq:+.4f} "
                      f"({'BEATS Turbo' if cmb < tq else 'loses'}) combined-current={cmb-cur:+.4f}", flush=True)
    print(f"[cvt] done -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
