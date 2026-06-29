"""tools/eval_vectorized_carekv_decode.py — VPhase E.

Before/after runtime comparison of CARE-KV correction: cached per-(h,t) loop
vs the P5 batched vectorized path. Same cells, same data; reports PPL, the
PPL delta vs the loop, K/V reads, runtime, speedup, peak GPU mem.

Cells:
  1. BaseQuant INT3
  2. uniform INT3 + CARE-KV (loop / cached)
  3. uniform INT3 + CARE-KV (vectorized)
  4. KIVI-style INT3 + CARE-KV (vectorized)
  5. KVQuant-style INT3 pre-RoPE + CARE-KV (vectorized)
"""
from __future__ import annotations
import argparse, csv, os, sys, time

import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer
from CARE_KV.care_kv.baselines import (BaseQuantAdapter, CAREKVAdapter,
                                       eval_ppl_wikitext)
from CARE_KV.care_kv.baselines.common import (DEVICE, measure_peak_gpu_mb,
                                              reset_peak_gpu)

MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def run_cell(adapter, tok, seq_len, n):
    reset_peak_gpu()
    t0 = time.perf_counter()
    m = adapter.setup_model(MID)
    ppl, ntok = eval_ppl_wikitext(m, tok, seq_len, n)
    dt = time.perf_counter() - t0
    stats = adapter.collect_debug_stats() if hasattr(adapter, "collect_debug_stats") else {}
    peak = measure_peak_gpu_mb()
    if hasattr(adapter, "teardown"): adapter.teardown()
    del m
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return dict(ppl=round(ppl, 6), k_reads=stats.get("k_reads", 0),
                v_reads=stats.get("v_reads", 0), runtime_s=round(dt, 1),
                peak_gpu_MB=round(peak, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--skip-loop", action="store_true",
                    help="skip the slow cached-loop baseline")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    cells = [("BaseQuant_INT3", BaseQuantAdapter(bits=3))]
    if not args.skip_loop:
        cells.append(("uniform_INT3_CAREKV_loop",
                      CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                                    max_pages=16, correction_impl="cached")))
    cells += [
        ("uniform_INT3_CAREKV_vectorized",
         CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                       max_pages=16, correction_impl="vectorized")),
        ("KIVI_INT3_CAREKV_vectorized",
         CAREKVAdapter(mode="fixed", bits=3, base_quantizer="kivi_style",
                       bits_k=3, bits_v=3, max_pages=16, correction_impl="vectorized")),
        ("KVQuantPreRoPE_INT3_CAREKV_vectorized",
         CAREKVAdapter(mode="fixed", bits=3, base_quantizer="kvquant_style",
                       k_store_mode="pre_rope", bits_k=3, bits_v=3, max_pages=16,
                       correction_impl="vectorized")),
    ]

    rows = []
    loop_ppl = loop_rt = None
    for name, adp in cells:
        r = run_cell(adp, tok, args.seq_len, args.num_samples)
        r["cell"] = name
        if name.endswith("_loop"):
            loop_ppl, loop_rt = r["ppl"], r["runtime_s"]
        rows.append(r)
        print(f"[VEC-E] {name:42s} PPL={r['ppl']:.5f} "
              f"K={r['k_reads']} V={r['v_reads']} rt={r['runtime_s']}s "
              f"peak={r['peak_gpu_MB']}MB", flush=True)

    # speedup + ppl diff vs loop (for the uniform vectorized cell)
    for r in rows:
        r["ppl_diff_vs_loop"] = ""
        r["speedup_vs_loop"] = ""
        if loop_ppl is not None and "uniform_INT3_CAREKV_vectorized" == r["cell"]:
            r["ppl_diff_vs_loop"] = round(abs(r["ppl"] - loop_ppl), 6)
            r["speedup_vs_loop"] = round(loop_rt / max(r["runtime_s"], 1e-9), 2)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = ["cell", "ppl", "ppl_diff_vs_loop", "k_reads", "v_reads",
            "runtime_s", "speedup_vs_loop", "peak_gpu_MB"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    if loop_ppl is not None:
        vec = next(r for r in rows if r["cell"] == "uniform_INT3_CAREKV_vectorized")
        print(f"\nuniform CARE-KV: loop PPL={loop_ppl} rt={loop_rt}s  →  "
              f"vectorized PPL={vec['ppl']} rt={vec['runtime_s']}s  "
              f"(Δppl={vec['ppl_diff_vs_loop']}, speedup={vec['speedup_vs_loop']}x)")
    print(f"wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
