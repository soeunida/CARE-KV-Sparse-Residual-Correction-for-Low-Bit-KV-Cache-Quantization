"""tools/eval_7b_validation.py — 7B CARE-KV validation via the vectorized path.

Confirms the TinyLlama characteristic (CARE-KV recovers BaseQuant-INT3 toward
fp16) transfers to a 7B model, now FEASIBLE because correction_impl=vectorized
(P5) removes the per-(layer,head,t) Python loop.

Arms (WikiText-2 PPL): fp16, base_quant_INT3, uniform+CARE-KV (vectorized,
SK2 SV4 RK2 RV2, sketch_dim default=32 full-rank).

Outputs: <out-csv>.
"""
from __future__ import annotations
import argparse, csv, os, sys, time

sys.path.insert(0, "/home/soeun")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from CARE_KV.care_kv.baselines import FP16Adapter, BaseQuantAdapter, CAREKVAdapter
from eval_base_quantizer_expansion import run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--model-id", default="deepseek-ai/deepseek-llm-7b-base")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-samples", type=int, default=4)
    args = ap.parse_args()
    maxp = args.seq_len // 16 + 8

    arms = [
        ("fp16", FP16Adapter()),
        ("base_int3", BaseQuantAdapter(bits=3)),
        ("carekv_uniform_vec", CAREKVAdapter(
            mode="fixed", bits=3, base_quantizer="uniform",
            sk=2, sv=4, rk=2, rv=2, max_pages=maxp,
            correction_impl="vectorized")),
    ]
    rows = []
    for label, ad in arms:
        t0 = time.perf_counter()
        r = run_one(ad, args.model_id, "wikitext", args.seq_len, args.num_samples)
        print(f"[7B-VAL] {label:20s} {ad.name:30s} PPL={r.ppl:9.4f} "
              f"K={r.k_reads} V={r.v_reads} resMB={r.residual_memory_MB:.3f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        d = r.as_dict(); d["arm"] = label
        rows.append(d)

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
