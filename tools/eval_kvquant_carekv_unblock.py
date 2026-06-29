"""tools/eval_kvquant_carekv_unblock.py

KVQuant-style + CARE-KV unblock evaluation.

Background: KVQuant's defining trait is **pre-RoPE K quantization** (the
unrotated per-channel K distribution is smoother). CARE-KV's cache stores
post-RoPE K and computes its residual in post-RoPE coordinates, so the
stacked "KVQuant-style + CARE-KV" cell used to be recorded as
`unsupported` (pre-RoPE vs post-RoPE residual coordinate mismatch).

This driver evaluates the unblocked path: K is quantized PRE-RoPE
(KVQuant-style), K_hat is re-rotated, and the CARE-KV residual is computed
in the post-RoPE coordinate system the correction reads. See
layer.py:prefill (CAREKV_K_STORE_MODE=pre_rope).

Cells (6):
 1. fp16                                            (reference)
 2. base_quant_INT3                                 (uniform INT3 baseline)
 3. KVQuant_style_INT3 (pre-RoPE, standalone)
 4. KVQuant_style_INT3 (pre-RoPE) + CARE-KV         (UNBLOCKED — this turn)
 5. KIVI_style_INT3 + CARE-KV                       (Phase Q-stacked ref)
 6. uniform_INT3 + CARE-KV                          (paper-best ref)

Outputs:
  results/.../ablations/kvquant_carekv_unblock_wt2_n4.csv
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
from CARE_KV.care_kv.baselines.kvquant_style import KVQuantStyleAdapter
from CARE_KV.care_kv.baselines.common import (
    DEVICE, measure_peak_gpu_mb, reset_peak_gpu,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build_adapters(max_pages: int) -> List[KVMethodAdapter]:
    return [
        FP16Adapter(),
        BaseQuantAdapter(bits=3),
        # KVQuant-style standalone, true pre-RoPE K.
        KVQuantStyleAdapter(bits_k=3, bits_v=3, k_store_mode="pre_rope"),
        # KVQuant-style pre-RoPE + CARE-KV — the unblocked stacked path.
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="kvquant_style",
                       k_store_mode="pre_rope", bits_k=3, bits_v=3,
                       max_pages=max_pages),
        # KIVI + CARE-KV (Phase Q-stacked reference).
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="kivi_style",
                       bits_k=3, bits_v=3, max_pages=max_pages),
        # uniform + CARE-KV (paper-best reference).
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                       max_pages=max_pages),
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
        model_id=model_id,
        dataset=dataset,
        seq_len=seq_len,
        num_samples=num_samples,
        bit_width=adapter.bit_width,
        k_quant_scheme=adapter.k_quant_scheme,
        v_quant_scheme=adapter.v_quant_scheme,
        uses_residual=adapter.uses_residual,
        uses_query_aware_routing=adapter.uses_query_aware_routing,
        notes=adapter.notes(),
    )
    if adapter.is_unsupported:
        return row

    # Shared GPU: external jobs can transiently consume all memory and OOM
    # our model load / eval. Retry a few times with backoff before giving up.
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
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print(f"[KVQ-UNBLOCK] OOM on {adapter.name} "
                      f"(attempt {attempt+1}/{max_retries}); "
                      f"waiting {retry_wait:.0f}s for GPU…", flush=True)
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
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        import traceback; traceback.print_exc()
        row.ppl = 0.0
        row.notes = f"ERROR: {type(e).__name__}: {e}"
        row.official_or_reimpl = "unsupported"
        if hasattr(adapter, "teardown"):
            try: adapter.teardown()
            except Exception: pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--dataset", default="wikitext",
                    choices=["synthetic", "wikitext"])
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-pages", type=int, default=16,
                    help="CARE-KV side-buffer pages (16 covers SL<=256 at page_size 16)")
    ap.add_argument("--only", default="",
                    help="substring filter on method name (smoke runs)")
    args = ap.parse_args()

    adapters = build_adapters(args.max_pages)
    if args.only:
        adapters = [a for a in adapters if args.only.lower() in a.name.lower()]
        if not adapters:
            print(f"no adapters match --only={args.only!r}", flush=True)
            sys.exit(2)

    rows: List[ResultRow] = []
    for a in adapters:
        r = run_one(a, MODEL_ID, args.dataset, args.seq_len, args.num_samples)
        rows.append(r)
        marker = "✗" if r.official_or_reimpl == "unsupported" else "✓"
        print(f"[KVQ-UNBLOCK] {marker} {a.name:52s} PPL={r.ppl:9.4f}  "
              f"mem≈{r.estimated_kv_memory_MB:6.2f}MB  "
              f"K_reads={r.k_reads:>7d} V_reads={r.v_reads:>7d}  "
              f"peak_gpu={r.peak_gpu_memory_MB:.0f}MB  ({r.runtime_seconds:.1f}s)",
              flush=True)

    fp16_ppl = next((r.ppl for r in rows
                     if r.method_name == "fp16" and r.ppl > 0), None)
    int3_ppl = next((r.ppl for r in rows
                     if r.method_name == "base_quant_INT3" and r.ppl > 0), None)
    for r in rows:
        if r.ppl > 0:
            if fp16_ppl is not None: r.dppl_vs_fp16 = round(r.ppl - fp16_ppl, 4)
            if int3_ppl is not None: r.dppl_vs_base_quant_int3 = round(r.ppl - int3_ppl, 4)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    dicts = [r.as_dict() for r in rows]
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
