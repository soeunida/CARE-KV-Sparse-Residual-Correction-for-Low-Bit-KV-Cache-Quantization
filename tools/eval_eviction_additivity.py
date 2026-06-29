"""tools/eval_eviction_additivity.py — CARE-KV ⊕ token-eviction additivity.

Tests the Section-2 orthogonality claim: CARE-KV's sparse residual correction is
ADDITIVE on top of token eviction (SnapKV/H2O-style). Eviction is applied to the
BASE attention (gated CAREKV_EVICT_KEEP_RATIO in layer.py), so the residual router
and correction operate on the kept set.

Four arms (all via CAREKVAdapter so the eviction hook applies uniformly):
  base_noevict   : INT3 base (READ budget 0), keep=1.0
  carekv_noevict : INT3 + CARE-KV SK2SV4, keep=1.0
  base_evict     : INT3 base, keep=R               (eviction only)
  carekv_evict   : INT3 + CARE-KV SK2SV4, keep=R   (eviction + CARE)
+ fp16 reference.

Additivity / orthogonality:
  CARE gain (no evict) = base_noevict - carekv_noevict   (>0)
  CARE gain (evict)    = base_evict   - carekv_evict      (>0)
  ⇒ if the two gains are similar, CARE-KV and eviction are ADDITIVE (orthogonal).

Run:
  CUDA_VISIBLE_DEVICES=2 python tools/eval_eviction_additivity.py \
    --out_csv results/eviction_additivity/evict_add.csv \
    --num-samples 4 --seq-len 512 --keep-ratio 0.5 --evict-policy h2o \
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

_EVICT_ENV = ["CAREKV_EVICT_KEEP_RATIO", "CAREKV_EVICT_POLICY", "CAREKV_EVICT_RECENT", "CAREKV_EVICT_SINK"]


def maxp_for(sl):
    return max(16, math.ceil(sl / 16) + 8)


def carekv(sk, sv, rk, rv, maxp):
    return CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform", k_store_mode="post_rope",
                         bits_k=3, bits_v=3, sk=sk, sv=sv, rk=rk, rv=rv, max_pages=maxp)


# arm -> (factory, keep_ratio_uses_R)
def arm_spec(arm, maxp):
    if arm == "fp16":           return FP16Adapter(), False
    if arm == "base_noevict":   return carekv(0, 0, 0, 0, maxp), False
    if arm == "carekv_noevict": return carekv(2, 4, 2, 2, maxp), False
    if arm == "base_evict":     return carekv(0, 0, 0, 0, maxp), True
    if arm == "carekv_evict":   return carekv(2, 4, 2, 2, maxp), True
    raise ValueError(arm)


ARMS = ["fp16", "base_noevict", "carekv_noevict", "base_evict", "carekv_evict"]
COLS = ["model_id", "seq_len", "num_samples", "keep_ratio", "evict_policy", "arm", "ppl",
        "k_reads", "v_reads", "runtime_s", "status", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--evict-policy", default="h2o")
    ap.add_argument("--evict-recent", type=int, default=0)   # 0 → layer default (N//10)
    ap.add_argument("--evict-sink", type=int, default=4)
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

    def set_evict(on):
        for k in _EVICT_ENV: os.environ.pop(k, None)
        if on:
            os.environ["CAREKV_EVICT_KEEP_RATIO"] = str(A.keep_ratio)
            os.environ["CAREKV_EVICT_POLICY"] = A.evict_policy
            os.environ["CAREKV_EVICT_SINK"] = str(A.evict_sink)
            if A.evict_recent > 0:
                os.environ["CAREKV_EVICT_RECENT"] = str(A.evict_recent)

    print(f"[evict-add] models={A.models} SL{A.seq_len} N{A.num_samples} keep={A.keep_ratio} "
          f"policy={A.evict_policy}", flush=True)
    for mid in A.models:
        maxp = maxp_for(A.seq_len)
        for arm in ARMS:
            if (mid, str(A.seq_len), arm) in done:
                print(f"[evict-add] skip {mid} {arm}", flush=True); continue
            ad, uses_R = arm_spec(arm, maxp)
            set_evict(uses_R)
            t0 = time.perf_counter()
            try:
                r = run_one(ad, mid, "wikitext", A.seq_len, A.num_samples)
                ppl = float(r.ppl) if (r.ppl and math.isfinite(float(r.ppl))) else float("nan")
                rec = dict(model_id=mid, seq_len=A.seq_len, num_samples=A.num_samples,
                           keep_ratio=A.keep_ratio if uses_R else 1.0, evict_policy=A.evict_policy if uses_R else "-",
                           arm=arm, ppl=round(ppl, 4) if math.isfinite(ppl) else "nan",
                           k_reads=r.k_reads, v_reads=r.v_reads,
                           runtime_s=round(time.perf_counter() - t0, 1),
                           status="real" if math.isfinite(ppl) else "nonfinite", notes="")
                print(f"[evict-add] {mid.split('/')[-1]:22s} {arm:16s} PPL={rec['ppl']}  "
                      f"K={r.k_reads} V={r.v_reads}  ({rec['runtime_s']}s)", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                rec = dict(model_id=mid, seq_len=A.seq_len, num_samples=A.num_samples, arm=arm,
                           status="error", notes=f"{type(e).__name__}: {str(e)[:80]}",
                           runtime_s=round(time.perf_counter() - t0, 1))
            for k in _EVICT_ENV: os.environ.pop(k, None)
            rows.append(rec); flush()
        # additivity summary print
        def g(a):
            for r in rows:
                if r["model_id"] == mid and r["arm"] == a:
                    try: return float(r["ppl"])
                    except Exception: return None
            return None
        bn, cn, be, ce = g("base_noevict"), g("carekv_noevict"), g("base_evict"), g("carekv_evict")
        if None not in (bn, cn, be, ce):
            print(f"[evict-add] {mid.split('/')[-1]}: CARE gain no-evict={bn-cn:+.3f}, "
                  f"with-evict={be-ce:+.3f}  → additive? "
                  f"{'YES' if abs((bn-cn)-(be-ce))<0.5*max(abs(bn-cn),0.1) else 'check'}", flush=True)
    print(f"[evict-add] done. {len(rows)} rows -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
