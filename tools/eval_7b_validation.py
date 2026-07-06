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
    ap.add_argument("--sketch-dim", type=int, default=0,
                    help=">0 overrides sketch_dim via CAREKV_SKETCH_DIM (attribution ablation)")
    ap.add_argument("--carekv-only", action="store_true",
                    help="skip fp16/base arms")
    ap.add_argument("--arm", choices=["fp16", "base", "carekv"], default=None,
                    help="run ONLY this arm (separate-process isolation for big "
                         "models); appends to --out-csv so 3 sequential runs "
                         "build the full table with fresh memory each")
    ap.add_argument("--bits", type=int, default=3, choices=[2, 3, 4],
                    help="base KV bit-width (2 = larger headroom for CARE-KV)")
    args = ap.parse_args()
    maxp = args.seq_len // 16 + 8
    if args.sketch_dim > 0:
        os.environ["CAREKV_SKETCH_DIM"] = str(args.sketch_dim)   # picked up by apply_carekv_env_overrides

    care_label = f"carekv_uniform_vec_sk{args.sketch_dim}" if args.sketch_dim > 0 else "carekv_uniform_vec"
    arms = [
        ("fp16", FP16Adapter()),
        (f"base_int{args.bits}", BaseQuantAdapter(bits=args.bits)),
        (care_label, CAREKVAdapter(
            mode="fixed", bits=args.bits, base_quantizer="uniform",
            sk=2, sv=4, rk=2, rv=2, max_pages=maxp,
            correction_impl="vectorized")),
    ]
    if args.carekv_only:
        arms = [a for a in arms if a[0].startswith("carekv")]
    if args.arm:
        want = {"fp16": "fp16", "base": "base_int", "carekv": "carekv"}[args.arm]
        arms = [a for a in arms if a[0].startswith(want)]
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
    # --arm isolation: append (write header only for a new file) so 3 sequential
    # single-arm processes accumulate into one CSV without a stale full-run header.
    append = bool(args.arm) and os.path.exists(args.out_csv)
    if append:
        import csv as _csv
        existing = list(_csv.DictReader(open(args.out_csv)))
        # drop any prior row for this same arm (idempotent re-run)
        existing = [r for r in existing if r.get("arm") not in {a[0] for a in arms}]
        for r in existing:
            for k in r:
                if k not in keys: keys.append(k)
        rows = existing + rows
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for d in rows: w.writerow(d)
    print(f"wrote {len(rows)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
