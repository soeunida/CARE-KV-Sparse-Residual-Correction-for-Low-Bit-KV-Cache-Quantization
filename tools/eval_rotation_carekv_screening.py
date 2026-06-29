"""tools/eval_rotation_carekv_screening.py

Screening stage for the rotation + CARE-KV stack. Tests whether a rotation-
improved base + CARE-KV sparse residual beats uniform + CARE-KV, and whether the
rotation must be pre-RoPE. Reuses the baselines KVMethodAdapter harness +
run_one from eval_base_quantizer_expansion.py.

Arms (--arms; default all):
  fp16, base_int3, uniform_carekv (bar),
  rot_post_carekv, rot_pre_carekv (arm4), rand_pre_carekv (arm5),
  rand_pre_base, rot_pre_base   (*_base = CARE budget 0 ⇒ rotation-only by the
                                 READ=0 ≡ base_quant invariant)

DIAGNOSTIC: TinyLlama N=4 SL=128. GO: rot/rand_pre_carekv beats uniform_carekv
by >0.02 with no KV-mem increase.
Outputs: <out-csv>.
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from typing import Dict, Callable

from CARE_KV.care_kv.baselines import (
    KVMethodAdapter, FP16Adapter, BaseQuantAdapter, CAREKVAdapter,
)

sys.path.insert(0, "/home/soeun")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from eval_base_quantizer_expansion import run_one

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAXP = 16


def _carekv(bq, kmode, sk=2, sv=4, rk=2, rv=2):
    return CAREKVAdapter(mode="fixed", bits=3, base_quantizer=bq,
                         k_store_mode=kmode, bits_k=3, bits_v=3,
                         sk=sk, sv=sv, rk=rk, rv=rv, max_pages=MAXP)


def _standalone(bq, kmode, label):
    a = _carekv(bq, kmode, sk=0, sv=0, rk=0, rv=0)
    a.name = label
    return a


ARMS: Dict[str, Callable[[], KVMethodAdapter]] = {
    "fp16":            lambda: FP16Adapter(),
    "base_int3":       lambda: BaseQuantAdapter(bits=3),
    "uniform_carekv":  lambda: _carekv("uniform", "post_rope"),
    "rot_post_carekv": lambda: _carekv("rotatekv_style", "post_rope"),
    "rot_pre_carekv":  lambda: _carekv("rotatekv_style", "pre_rope"),
    "rand_pre_carekv": lambda: _carekv("randrot_style", "pre_rope"),
    "rand_pre_base":   lambda: _standalone("randrot_style", "pre_rope",
                                           "RandRot_preRoPE_INT3_standalone"),
    "rot_pre_base":    lambda: _standalone("rotatekv_style", "pre_rope",
                                           "RotateKV_preRoPE_INT3_standalone"),
}
DEFAULT_ARMS = list(ARMS.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    args = ap.parse_args()

    chosen = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = [a for a in chosen if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arms: {bad}. known: {list(ARMS)}")

    rows = []
    for arm in chosen:
        adapter = ARMS[arm]()
        t0 = time.perf_counter()
        r = run_one(adapter, MODEL_ID, "wikitext", args.seq_len, args.num_samples)
        marker = "x" if r.official_or_reimpl == "unsupported" else "ok"
        print(f"[ROT-SCREEN] {marker} {arm:16s} {r.method_name:42s} "
              f"PPL={r.ppl:9.4f}  resMB={r.residual_memory_MB:6.3f}  "
              f"K_reads={r.k_reads:>8d} V_reads={r.v_reads:>8d}  "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        d = r.as_dict(); d["arm"] = arm
        rows.append(d)

    bar = next((d["ppl"] for d in rows if d.get("arm") == "uniform_carekv"
                and float(d.get("ppl") or 0) > 0), None)
    fp16 = next((d["ppl"] for d in rows if d.get("arm") == "fp16"
                 and float(d.get("ppl") or 0) > 0), None)
    for d in rows:
        p = float(d.get("ppl") or 0)
        d["dppl_vs_uniform_carekv"] = (round(p - float(bar), 4)
                                        if (bar and p > 0) else "")
        d["dppl_vs_fp16"] = (round(p - float(fp16), 4)
                              if (fp16 and p > 0) else "")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = []
    for d in rows:
        for k in d:
            if k not in keys: keys.append(k)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for d in rows: w.writerow(d)
    print(f"wrote {len(rows)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
