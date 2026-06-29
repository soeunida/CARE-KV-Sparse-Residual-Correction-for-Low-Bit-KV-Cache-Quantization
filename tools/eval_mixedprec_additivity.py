"""tools/eval_mixedprec_additivity.py — CARE-KV ⊕ mixed-precision additivity.

Tests the Section-2 orthogonality claim: CARE-KV's sparse residual correction is
ADDITIVE on top of mixed-precision (LeanKV/MiKV-style) base quantization. Mixed
precision is applied to the base K̂/V̂ (gated CAREKV_MIXEDPREC_HI_FRAC in layer.py):
salient tokens kept at bits_hi, the rest at bits_lo (avg ≈ INT3), so the residual is
computed against the mixed base and CARE-KV stacks on top.

Five arms (all via CAREKVAdapter so the mixed-precision hook applies uniformly):
  base_uniform     : INT3 uniform base (READ 0), no mixedprec
  carekv_uniform   : INT3 uniform + CARE-KV SK2SV4
  base_mixedprec   : mixed-precision base (READ 0)              [mixedprec only]
  carekv_mixedprec : mixed-precision base + CARE-KV SK2SV4      [mixedprec + CARE]
+ fp16 reference.

Additivity / orthogonality:
  CARE gain (uniform)   = base_uniform   - carekv_uniform   (>0)
  CARE gain (mixedprec) = base_mixedprec - carekv_mixedprec (>0)
  ⇒ similar gains → CARE-KV and mixed-precision are ADDITIVE (orthogonal).

Run:
  CUDA_VISIBLE_DEVICES=2 python tools/eval_mixedprec_additivity.py \
    --out_csv results/mixedprec_additivity/mp_tinyllama.csv \
    --num-samples 4 --seq-len 256 --hi-frac 0.5 --bits-hi 4 --bits-lo 2 \
    --models TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""
import os, sys, csv, time, math, argparse
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/CARE_KV/care_kv/tools")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("CAREKV_VECTORIZED_RESIDUAL", "1")
os.environ.setdefault("CAREKV_VECTORIZE_VDOM_ONLY", "1")

import torch
from CARE_KV.care_kv.baselines import FP16Adapter, CAREKVAdapter
from eval_base_quantizer_expansion import run_one

_MP_ENV = ["CAREKV_MIXEDPREC_HI_FRAC", "CAREKV_MIXEDPREC_BITS_HI",
           "CAREKV_MIXEDPREC_BITS_LO", "CAREKV_MIXEDPREC_SALIENCY"]


def maxp_for(sl):
    return max(16, math.ceil(sl / 16) + 8)


def carekv(sk, sv, rk, rv, maxp):
    return CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform", k_store_mode="post_rope",
                         bits_k=3, bits_v=3, sk=sk, sv=sv, rk=rk, rv=rv, max_pages=maxp)


def arm_spec(arm, maxp):
    if arm == "fp16":             return FP16Adapter(), False
    if arm == "base_uniform":     return carekv(0, 0, 0, 0, maxp), False
    if arm == "carekv_uniform":   return carekv(2, 4, 2, 2, maxp), False
    if arm == "base_mixedprec":   return carekv(0, 0, 0, 0, maxp), True
    if arm == "carekv_mixedprec": return carekv(2, 4, 2, 2, maxp), True
    raise ValueError(arm)


ARMS = ["fp16", "base_uniform", "carekv_uniform", "base_mixedprec", "carekv_mixedprec"]
COLS = ["model_id", "seq_len", "num_samples", "hi_frac", "bits_hi", "bits_lo", "saliency",
        "arm", "ppl", "k_reads", "v_reads", "runtime_s", "status", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--hi-frac", type=float, default=0.5)
    ap.add_argument("--bits-hi", type=int, default=4)
    ap.add_argument("--bits-lo", type=int, default=2)
    ap.add_argument("--saliency", default="vnorm")
    ap.add_argument("--models", nargs="+", required=True)
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

    def set_mp(on):
        for k in _MP_ENV: os.environ.pop(k, None)
        if on:
            os.environ["CAREKV_MIXEDPREC_HI_FRAC"] = str(A.hi_frac)
            os.environ["CAREKV_MIXEDPREC_BITS_HI"] = str(A.bits_hi)
            os.environ["CAREKV_MIXEDPREC_BITS_LO"] = str(A.bits_lo)
            os.environ["CAREKV_MIXEDPREC_SALIENCY"] = A.saliency

    avg_bits = A.hi_frac * A.bits_hi + (1 - A.hi_frac) * A.bits_lo
    print(f"[mp-add] models={A.models} SL{A.seq_len} N{A.num_samples} hi_frac={A.hi_frac} "
          f"bits={A.bits_hi}/{A.bits_lo} (avg={avg_bits:.2f}) sal={A.saliency}", flush=True)
    for mid in A.models:
        maxp = maxp_for(A.seq_len)
        for arm in ARMS:
            if (mid, str(A.seq_len), arm) in done:
                print(f"[mp-add] skip {mid} {arm}", flush=True); continue
            ad, uses_mp = arm_spec(arm, maxp)
            set_mp(uses_mp)
            t0 = time.perf_counter()
            try:
                r = run_one(ad, mid, "wikitext", A.seq_len, A.num_samples)
                ppl = float(r.ppl) if (r.ppl and math.isfinite(float(r.ppl))) else float("nan")
                rec = dict(model_id=mid, seq_len=A.seq_len, num_samples=A.num_samples,
                           hi_frac=A.hi_frac if uses_mp else "-", bits_hi=A.bits_hi if uses_mp else "-",
                           bits_lo=A.bits_lo if uses_mp else "-", saliency=A.saliency if uses_mp else "-",
                           arm=arm, ppl=round(ppl, 4) if math.isfinite(ppl) else "nan",
                           k_reads=r.k_reads, v_reads=r.v_reads,
                           runtime_s=round(time.perf_counter() - t0, 1),
                           status="real" if math.isfinite(ppl) else "nonfinite", notes="")
                print(f"[mp-add] {mid.split('/')[-1]:22s} {arm:18s} PPL={rec['ppl']}  "
                      f"K={r.k_reads} V={r.v_reads}  ({rec['runtime_s']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                rec = dict(model_id=mid, seq_len=A.seq_len, num_samples=A.num_samples, arm=arm,
                           status="error", notes=f"{type(e).__name__}: {str(e)[:80]}",
                           runtime_s=round(time.perf_counter() - t0, 1))
            for k in _MP_ENV: os.environ.pop(k, None)
            rows.append(rec); flush()

        def g(a):
            for r in rows:
                if r["model_id"] == mid and r["arm"] == a:
                    try: return float(r["ppl"])
                    except Exception: return None
            return None
        bu, cu, bm, cm = g("base_uniform"), g("carekv_uniform"), g("base_mixedprec"), g("carekv_mixedprec")
        if None not in (bu, cu, bm, cm):
            g0, g1 = bu - cu, bm - cm
            add = "YES" if (g0 > 0 and g1 > 0 and abs(g0 - g1) < 0.6 * max(g0, 0.1)) else "check"
            print(f"[mp-add] {mid.split('/')[-1]}: CARE gain uniform={g0:+.3f}, "
                  f"mixedprec={g1:+.3f}  → additive? {add}", flush=True)
    print(f"[mp-add] done. {len(rows)} rows -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
