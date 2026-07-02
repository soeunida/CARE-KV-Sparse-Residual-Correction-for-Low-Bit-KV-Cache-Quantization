"""tools/eval_router_prefilter_sweep.py — Phase-1 C sweep for the router
O(S) scoring-BW pre-filter (router_prefilter_bw_design.md).

For each shortlist size C (CAREKV_ROUTER_PREFILTER_C), run CARE-KV (vectorized,
paper-best, sketch_dim=32) and report WikiText-2 PPL + the fraction of K
candidates that were exact-scored (scored/pool) — the analytical sketch-read
reduction. C=0 = exact / no pre-filter (invariant reference).

At long context the K candidate pool = ~n_pages·SK, so C≪pool is the win.
Outputs: <out-csv>.
"""
from __future__ import annotations
import argparse, csv, os, sys, time

sys.path.insert(0, "/home/soeun")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoTokenizer
import diag_router_bottleneck as D
from CARE_KV.care_kv import get_debug_stats
from CARE_KV.care_kv.baselines.common import eval_ppl_wikitext, DEVICE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--cs", default="0,4,8,16,32,64")
    ap.add_argument("--sign-b", type=int, default=0,
                    help="sign-sketch proxy bits (0 = v1 magnitude bound)")
    args = ap.parse_args()
    Cs = [int(x) for x in args.cs.split(",") if x.strip() != ""]
    os.environ["CAREKV_ROUTER_SIGN_PREFILTER_B"] = str(args.sign_b)
    mp = max(16, args.seq_len // 16 + 8)
    tok = AutoTokenizer.from_pretrained(D.MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    base_ppl = None
    rows = []
    for C in Cs:
        os.environ["CAREKV_ROUTER_PREFILTER_C"] = str(C)
        t0 = time.perf_counter()
        try:
            m = D.build("carekv_stored", sketch_dim=32, sk=2, sv=4, rk=2, rv=2,
                        corr_impl="vectorized", max_pages=mp)
            ppl, ntok = eval_ppl_wikitext(m, tok, args.seq_len, args.num_samples)
            st = get_debug_stats()
            pool = int(st.get("k_prefilter_pool", 0))
            scored = int(st.get("k_prefilter_scored", 0))
            frac = round(scored / pool, 4) if pool else 1.0
            if C == 0:
                base_ppl = ppl
            row = dict(prefilter_C=C, sign_b=args.sign_b, ppl=round(float(ppl), 4),
                       dppl_vs_exact=("" if base_ppl is None else round(ppl - base_ppl, 4)),
                       k_prefilter_pool=pool, k_prefilter_scored=scored,
                       scored_frac=frac,
                       stage1_B_per_cand=round(float(st.get("k_stage1_bytes_per_cand", 0)), 2),
                       k_reads=int(st.get("k_slots_read", 0)),
                       seconds=round(time.perf_counter() - t0, 1))
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            row = dict(prefilter_C=C, ppl=0.0, error=f"{type(e).__name__}: {e}")
        rows.append(row)
        print(f"[PF] C={C:<3d} PPL={row.get('ppl',0):9.4f} "
              f"dvsExact={row.get('dppl_vs_exact','')} "
              f"scored/pool={row.get('scored_frac','?')} "
              f"({row.get('seconds','?')}s)", flush=True)

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
