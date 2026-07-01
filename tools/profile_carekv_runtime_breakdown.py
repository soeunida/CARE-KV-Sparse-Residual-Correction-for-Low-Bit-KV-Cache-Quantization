"""tools/profile_carekv_runtime_breakdown.py — VPhase A.

Profile where CARE-KV prefill/decode wall-clock goes, to target the
vectorization work. Wraps the hot functions with timing + call-count
accumulators (monkeypatch in THIS process only — does not affect any other
running job) and runs one small real setting:

  TinyLlama-1.1B, WikiText-2, SL=128, N=1, base_bits=3, carekv_stored,
  SK=2 SV=4 RK=2 RV=2.

Buckets:
  - model_forward_total      (whole model() call)
  - carekv_prefill           (CAREKVLayer.prefill, all layers)
  - router_scoring           (ResidualRouter.route)
  - correction_cached        (apply_slot_corrections, per-(h,t) loop)
  - correction_vectorized_V  (vectorized_v_correction)
  - sparse_correction_driver (_apply_sparse_prefill_correction_stored)
plus call counts (= Python-loop iteration counts) and K_reads/V_reads.

Outputs:
  results/.../ablations/carekv_runtime_breakdown_before.csv
  results/.../summaries/carekv_runtime_breakdown_before.md  (written by caller)
"""
from __future__ import annotations
import argparse, csv, os, sys, time, functools

import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

_ACC = {}   # name -> [total_seconds, call_count]


def _timed(name, fn):
    @functools.wraps(fn)
    def wrap(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            rec = _ACC.setdefault(name, [0.0, 0])
            rec[0] += time.perf_counter() - t0
            rec[1] += 1
    return wrap


def install_timers():
    """Wrap hot functions. Returns list of (obj, attr, original) to restore."""
    patches = []
    from CARE_KV.care_kv import attention, layer, residual_router

    def patch(obj, attr, label):
        orig = getattr(obj, attr, None)
        if orig is None:
            return
        setattr(obj, attr, _timed(label, orig))
        patches.append((obj, attr, orig))

    patch(residual_router.ResidualRouter, "route", "router_scoring")
    patch(attention, "apply_slot_corrections", "correction_cached")
    if hasattr(attention, "vectorized_v_correction"):
        patch(attention, "vectorized_v_correction", "correction_vectorized_V")
    patch(layer.CAREKVLayer, "prefill", "carekv_prefill")
    if hasattr(layer.CAREKVLayer, "_apply_sparse_prefill_correction_stored"):
        patch(layer.CAREKVLayer, "_apply_sparse_prefill_correction_stored",
              "sparse_correction_driver")
    # base quantization (best-effort — name may vary)
    for cand in ("quant_dequant", "quantize_dequantize", "fake_quant"):
        if hasattr(attention, cand):
            patch(attention, cand, "base_quant"); break
    return patches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=1)
    args = ap.parse_args()

    patches = install_timers()
    from transformers import AutoTokenizer
    from CARE_KV.care_kv.baselines import CAREKVAdapter, eval_ppl_wikitext
    from CARE_KV.care_kv.baselines.common import DEVICE, measure_peak_gpu_mb, reset_peak_gpu
    from CARE_KV.care_kv import get_debug_stats, reset_debug_stats

    MID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(MID)
    a = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                      sk=2, sv=4, rk=2, rv=2,
                      max_pages=(args.seq_len + 15) // 16 + 8)
    reset_debug_stats()
    reset_peak_gpu()
    m = a.setup_model(MID)

    # time the whole forward via the eval helper
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    ppl, ntok = eval_ppl_wikitext(m, tok, args.seq_len, args.num_samples)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    total = time.perf_counter() - t0
    _ACC["model_forward_total"] = [total, 1]

    stats = a.collect_debug_stats()
    peak = measure_peak_gpu_mb()

    # Build rows
    order = ["model_forward_total", "carekv_prefill", "sparse_correction_driver",
             "router_scoring", "correction_cached", "correction_vectorized_V",
             "base_quant"]
    rows = []
    for name in order:
        if name in _ACC:
            sec, cnt = _ACC[name]
            rows.append(dict(stage=name, total_s=round(sec, 3),
                             pct_of_forward=round(100 * sec / max(total, 1e-9), 2),
                             call_count=cnt,
                             ms_per_call=round(1000 * sec / max(cnt, 1), 3)))
    # extra rows
    rows.append(dict(stage="ppl", total_s=round(ppl, 4), pct_of_forward="",
                     call_count="", ms_per_call=""))
    rows.append(dict(stage="k_reads", total_s=stats["k_reads"], pct_of_forward="",
                     call_count="", ms_per_call=""))
    rows.append(dict(stage="v_reads", total_s=stats["v_reads"], pct_of_forward="",
                     call_count="", ms_per_call=""))
    rows.append(dict(stage="peak_gpu_MB", total_s=round(peak, 1), pct_of_forward="",
                     call_count="", ms_per_call=""))
    rows.append(dict(stage="correction_cached_call_count_=_Hq*T_loop_iters",
                     total_s="", pct_of_forward="",
                     call_count=_ACC.get("correction_cached", [0, 0])[1], ms_per_call=""))

    print("=== CARE-KV runtime breakdown (TinyLlama SL=%d N=%d) ==="
          % (args.seq_len, args.num_samples))
    for r in rows:
        if r["pct_of_forward"] != "":
            print(f"  {r['stage']:34s} {r['total_s']:8}s  {r['pct_of_forward']:5}%  "
                  f"calls={r['call_count']}  {r['ms_per_call']}ms/call")
    print(f"  PPL={ppl:.4f}  K_reads={stats['k_reads']}  V_reads={stats['v_reads']}  "
          f"peak={peak:.0f}MB")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "total_s", "pct_of_forward",
                                          "call_count", "ms_per_call"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote -> {args.out_csv}")

    for obj, attr, orig in patches:
        setattr(obj, attr, orig)


if __name__ == "__main__":
    main()
