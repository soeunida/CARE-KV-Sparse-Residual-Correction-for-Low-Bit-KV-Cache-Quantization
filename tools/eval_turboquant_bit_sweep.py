"""tools/eval_turboquant_bit_sweep.py — Part A: TurboQuant INT4/3/2 bit sweep.

Same-condition direct comparison of base quantizers across bit-widths,
with the TurboQuant-style reimplementation (random rotation + per-coord
scalar quant + QJL 1-bit residual inner-product correction). See
`baselines/turboquant_style.py` for the honest framing (NOT official).

Cells:
  1. fp16                                         (reference)
  2-4. BaseQuant INT4 / INT3 / INT2               (uniform group=32 base)
  5-7. TurboQuant_style INT4 / INT3 / INT2 (+QJL) (reimpl)
       + noQJL ablation rows (rotation+quant only) for each bit-width
  8-10. TurboQuant_style INT4/3/2 + CARE-KV       (UNSUPPORTED — see notes)
  11. uniform INT3 + CARE-KV                       (paper-best anchor)
  12. KIVI-style INT3 + CARE-KV                    (Phase Q anchor)
  13. KVQuant-style INT3 pre-RoPE + CARE-KV        (unblock anchor)

Outputs:
  results/.../ablations/turboquant_bit_sweep_wt2_n4.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from typing import List

import torch

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer
from CARE_KV.care_kv.baselines import (
    KVMethodAdapter, ResultRow,
    eval_ppl_wikitext, eval_ppl_synthetic,
    FP16Adapter, BaseQuantAdapter, CAREKVAdapter,
)
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from CARE_KV.care_kv.baselines.common import (
    DEVICE, measure_peak_gpu_mb, reset_peak_gpu, fp16_kv_mb,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class TurboQuantPlusCAREKVUnsupported(KVMethodAdapter):
    """TurboQuant + CARE-KV is unsupported: QJL is a *score-level inner-product*
    correction, not a *reconstruction* base quantizer, so CARE-KV's residual
    slots (which correct k_hat/v_hat reconstruction) cannot be stacked on the
    QJL estimate without redefining the method. Marked unsupported rather than
    inventing a stacking semantics. (uniform/KIVI/KVQuant bases ARE reconstruction
    quantizers and DO stack — see anchor cells.)"""
    family = "turboquant_plus_carekv"
    is_unsupported = True
    is_reimplementation = False

    def __init__(self, bits: int = 3):
        self.bits = bits
        self.name = f"TurboQuant_style_INT{bits}_plus_CAREKV"
        self.bit_width = f"INT{bits} + CARE-KV residual"
        self.k_quant_scheme = "TurboQuant QJL (score-level) — not stackable as reconstruction base"
        self.v_quant_scheme = "TurboQuant QJL (score-level) — not stackable as reconstruction base"
        self.unsupported_reason = (
            "TurboQuant's distinctive stage is QJL, a 1-bit residual *inner-product* "
            "estimator applied at attention-score time — not a reconstruction quantizer. "
            "CARE-KV residual slots correct k_hat/v_hat reconstruction, so they cannot "
            "be stacked on the QJL score estimate without redefining both methods. "
            "Unsupported this turn (no invented semantics). The rotation+per-coord-quant "
            "*reconstruction* part alone would duplicate RotateKV+CARE-KV.")

    def notes(self) -> str:
        return "UNSUPPORTED: " + self.unsupported_reason


def build_adapters(max_pages: int, qjl_m: int) -> List[KVMethodAdapter]:
    qm = qjl_m if qjl_m > 0 else 0
    return [
        FP16Adapter(),
        BaseQuantAdapter(bits=4),
        BaseQuantAdapter(bits=3),
        BaseQuantAdapter(bits=2),
        # TurboQuant-style standalone, +QJL (headline) and noQJL ablation.
        TurboQuantStyleAdapter(bits_k=4, bits_v=4, qjl_m=qm, use_qjl=True),
        TurboQuantStyleAdapter(bits_k=4, bits_v=4, qjl_m=qm, use_qjl=False),
        TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=qm, use_qjl=True),
        TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=qm, use_qjl=False),
        TurboQuantStyleAdapter(bits_k=2, bits_v=2, qjl_m=qm, use_qjl=True),
        TurboQuantStyleAdapter(bits_k=2, bits_v=2, qjl_m=qm, use_qjl=False),
        # TurboQuant + CARE-KV (unsupported stubs).
        TurboQuantPlusCAREKVUnsupported(bits=4),
        TurboQuantPlusCAREKVUnsupported(bits=3),
        TurboQuantPlusCAREKVUnsupported(bits=2),
        # CARE-KV anchors (reconstruction bases + CARE-KV residual).
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform", max_pages=max_pages),
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="kivi_style",
                       bits_k=3, bits_v=3, max_pages=max_pages),
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="kvquant_style",
                       k_store_mode="pre_rope", bits_k=3, bits_v=3, max_pages=max_pages),
    ]


def run_one(adapter: KVMethodAdapter, model_id: str,
             dataset: str, seq_len: int, num_samples: int) -> ResultRow:
    row = ResultRow(
        method_name=adapter.name,
        method_family=adapter.family,
        official_or_reimpl=(
            "unsupported" if adapter.is_unsupported else
            "same-condition reimplementation" if adapter.is_reimplementation else
            "reference"),
        model_id=model_id, dataset=dataset, seq_len=seq_len, num_samples=num_samples,
        bit_width=adapter.bit_width,
        k_quant_scheme=adapter.k_quant_scheme, v_quant_scheme=adapter.v_quant_scheme,
        uses_residual=adapter.uses_residual,
        uses_query_aware_routing=adapter.uses_query_aware_routing,
        notes=adapter.notes(),
    )
    if adapter.is_unsupported:
        return row

    max_retries = int(os.environ.get("KVQ_OOM_RETRIES", "6"))
    retry_wait = float(os.environ.get("KVQ_OOM_WAIT", "30"))
    t0 = time.perf_counter()
    reset_peak_gpu()
    try:
        last_oom = None
        for attempt in range(max_retries):
            try:
                m = adapter.setup_model(model_id)
                tok = AutoTokenizer.from_pretrained(model_id)
                if tok.pad_token_id is None:
                    tok.pad_token_id = tok.eos_token_id or 0
                if dataset == "wikitext":
                    ppl, n_tok = eval_ppl_wikitext(m, tok, seq_len, num_samples)
                else:
                    ppl, n_tok = eval_ppl_synthetic(m, tok, seq_len)
                break
            except torch.cuda.OutOfMemoryError as oom:
                last_oom = oom
                if hasattr(adapter, "teardown"):
                    try: adapter.teardown()
                    except Exception: pass
                try: del m
                except Exception: pass
                if DEVICE == "cuda": torch.cuda.empty_cache()
                print(f"[TQ-SWEEP] OOM on {adapter.name} "
                      f"(attempt {attempt+1}/{max_retries}); waiting {retry_wait:.0f}s…",
                      flush=True)
                time.sleep(retry_wait)
        else:
            raise last_oom
        dt = time.perf_counter() - t0
        peak = measure_peak_gpu_mb()
        stats = adapter.collect_debug_stats()
        mem = adapter.estimate_memory(seq_len)
        budgets = adapter.effective_budgets()

        row.ppl = round(ppl, 4)
        row.evaluated_tokens = int(n_tok)
        row.runtime_seconds = round(dt, 1)
        row.peak_gpu_memory_MB = round(peak, 1)
        row.estimated_kv_memory_MB = mem.get("estimated_kv_memory_MB", 0.0)
        row.estimated_total_cache_memory_MB = mem.get("estimated_total_cache_memory_MB", 0.0)
        row.vs_fp16_kv_memory_ratio = mem.get("vs_fp16_kv_memory_ratio", 1.0)
        row.base_memory_MB = mem.get("base_memory_MB", mem.get("estimated_kv_memory_MB", 0.0))
        row.residual_memory_MB = mem.get("residual_memory_MB", 0.0)
        row.base_quantizer = mem.get("base_quantizer", getattr(adapter, "base_quantizer", ""))
        row.k_reads = stats.get("k_reads", 0)
        row.v_reads = stats.get("v_reads", 0)
        row.stored_k_slots = stats.get("stored_k_slots", 0)
        row.stored_v_slots = stats.get("stored_v_slots", 0)
        row.effective_store_budget = budgets.get("effective_store_budget", "")
        row.effective_read_budget = budgets.get("effective_read_budget", "")
        if hasattr(adapter, "teardown"):
            try: adapter.teardown()
            except Exception: pass
        del m
        if DEVICE == "cuda": torch.cuda.empty_cache()
    except Exception as e:
        import traceback; traceback.print_exc()
        row.ppl = 0.0
        row.notes = f"ERROR: {type(e).__name__}: {e}"
        row.official_or_reimpl = "unsupported"
        if hasattr(adapter, "teardown"):
            try: adapter.teardown()
            except Exception: pass
    return row


def _bits_of(name: str):
    import re
    m = re.search(r"INT(\d)", name)
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--dataset", default="wikitext", choices=["synthetic", "wikitext"])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-pages", type=int, default=16)
    ap.add_argument("--qjl-m", type=int, default=0, help="QJL projection dim (0=2*head_dim)")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    adapters = build_adapters(args.max_pages, args.qjl_m)
    if args.only:
        adapters = [a for a in adapters if args.only.lower() in a.name.lower()]
        if not adapters:
            print(f"no adapters match --only={args.only!r}", flush=True); sys.exit(2)

    rows: List[ResultRow] = []
    for a in adapters:
        r = run_one(a, MODEL_ID, args.dataset, args.seq_len, args.num_samples)
        rows.append(r)
        marker = "✗" if r.official_or_reimpl == "unsupported" else "✓"
        print(f"[TQ-SWEEP] {marker} {a.name:42s} PPL={r.ppl:9.4f}  "
              f"mem≈{r.estimated_kv_memory_MB:6.2f}MB res≈{r.residual_memory_MB:5.2f}MB "
              f"K_reads={r.k_reads:>7d} V_reads={r.v_reads:>7d}  "
              f"peak={r.peak_gpu_memory_MB:.0f}MB  ({r.runtime_seconds:.1f}s)", flush=True)

    fp16_ppl = next((r.ppl for r in rows if r.method_name == "fp16" and r.ppl > 0), None)
    int3_ppl = next((r.ppl for r in rows if r.method_name == "base_quant_INT3" and r.ppl > 0), None)
    basequant_by_bits = {_bits_of(r.method_name): r.ppl for r in rows
                         if r.method_family == "base_quant" and r.ppl > 0}
    for r in rows:
        if r.ppl > 0:
            if fp16_ppl is not None: r.dppl_vs_fp16 = round(r.ppl - fp16_ppl, 4)
            if int3_ppl is not None: r.dppl_vs_base_quant_int3 = round(r.ppl - int3_ppl, 4)
            b = _bits_of(r.method_name)
            if b in basequant_by_bits:
                # store ΔPPL vs same-bit BaseQuant in effective_read_budget-adjacent free field via notes? No:
                r.metadata_dppl_vs_same_bit_basequant = round(r.ppl - basequant_by_bits[b], 4)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    dicts = []
    for r in rows:
        d = r.as_dict()
        d["dppl_vs_same_bit_basequant"] = getattr(r, "metadata_dppl_vs_same_bit_basequant", "")
        dicts.append(d)
    keys = []
    for d in dicts:
        for k in d:
            if k not in keys: keys.append(k)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for d in dicts: w.writerow(d)
    print(f"wrote {len(rows)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
