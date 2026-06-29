"""tools/eval_base_quantizer_expansion.py

Base-quantizer expansion beyond Phase Q's KIVI integration: adds
KVQuant-style + RotateKV-style same-condition reimplementations, and
records TurboQuant as `unsupported` with the blocker documented in
`summaries/turboquant_integration_status.md`.

Cells (same model / dataset / tokenizer / eval as the rest of the
Phase Q + P-direct sweeps so PPL is comparable across files):

 1. fp16                                            (reference)
 2. base_quant_INT4                                 (uniform INT4 ceiling)
 3. base_quant_INT3                                 (uniform INT3 baseline)
 4. uniform_INT3 + CARE-KV  (paper-best)
 5. KIVI_style_INT3                                 (PyTorch reimpl from baselines)
 6. KIVI_INT3 + CARE-KV (stacked)                   (Phase Q-stacked)
 7. KVQuant_style_INT3 (pre-RoPE K)
 8. KVQuant_style_INT3 + CARE-KV                    (unsupported — pre-RoPE blocker)
 9. RotateKV_style_INT3
10. RotateKV_INT3 + CARE-KV                          (NEW stacked)
11. TurboQuant_style_INT3                            (unsupported — see plan doc)
12. TurboQuant_INT3 + CARE-KV                        (unsupported — same blocker)

Outputs:
  results/.../ablations/base_quantizer_expansion_wt2_n4.csv
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
    FP16Adapter, BaseQuantAdapter, CAREKVAdapter, KIVIStyleAdapter,
)
from CARE_KV.care_kv.baselines.kvquant_style import (
    KVQuantStyleAdapter, KVQuantPlusCAREKVUnsupported,
)
from CARE_KV.care_kv.baselines.rotatekv_style import RotateKVStyleAdapter
from CARE_KV.care_kv.baselines.common import (
    DEVICE, measure_peak_gpu_mb, reset_peak_gpu,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ─────────────────────────────────────────────
# TurboQuant unsupported stubs (per integration plan)
# ─────────────────────────────────────────────

class TurboQuantStyleUnsupported(KVMethodAdapter):
    family = "turboquant_style"
    is_official = False
    is_reimplementation = False
    is_unsupported = True
    bit_width = "INT3 (per coordinate)"
    k_quant_scheme = "random rotation + per-coord scalar + 1-bit QJL residual"
    v_quant_scheme = "random rotation + per-coord scalar + 1-bit QJL residual"
    unsupported_reason = (
        "TurboQuant (arXiv 2504.19874): no official Google code release "
        "yet (Q2 2026 expected); four community implementations exist "
        "but differ on the 1-bit Quantized JL (QJL) residual specifics. "
        "Per spec we do NOT invent unclear TurboQuant behavior. See "
        "summaries/turboquant_integration_status.md for the full "
        "blocker writeup."
    )

    def __init__(self, bits: int = 3):
        self.bits = bits
        self.name = f"TurboQuant_style_INT{bits}"

    def setup_model(self, model_id: str):
        raise NotImplementedError(self.unsupported_reason)

    def notes(self) -> str:
        return self.unsupported_reason


class TurboQuantPlusCAREKVUnsupported(KVMethodAdapter):
    family = "turboquant_plus_carekv"
    is_official = False
    is_reimplementation = False
    is_unsupported = True
    bit_width = "INT3 + CARE-KV residual"
    k_quant_scheme = "TurboQuant + CARE-KV residual slots"
    v_quant_scheme = "TurboQuant + CARE-KV residual slots"
    uses_residual = True
    uses_query_aware_routing = True
    unsupported_reason = (
        "TurboQuant + CARE-KV stacked is blocked by the same TurboQuant "
        "feasibility blockers (no official code, community QJL specs "
        "disagree). See summaries/turboquant_integration_status.md."
    )

    def __init__(self, bits: int = 3):
        self.bits = bits
        self.name = f"TurboQuant_INT{bits}_plus_CAREKV"

    def setup_model(self, model_id: str):
        raise NotImplementedError(self.unsupported_reason)

    def notes(self) -> str:
        return self.unsupported_reason


# ─────────────────────────────────────────────
# Cell list (12 cells)
# ─────────────────────────────────────────────

def build_adapters() -> List[KVMethodAdapter]:
    return [
        # references / baselines
        FP16Adapter(),
        BaseQuantAdapter(bits=4),
        BaseQuantAdapter(bits=3),
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                       max_pages=16),
        # KIVI (completed reference from Phase Q-stacked)
        KIVIStyleAdapter(bits_k=3, bits_v=3),
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="kivi_style",
                       bits_k=3, bits_v=3, max_pages=16),
        # KVQuant
        KVQuantStyleAdapter(bits_k=3, bits_v=3, k_store_mode="pre_rope"),
        KVQuantPlusCAREKVUnsupported(bits=3),
        # RotateKV
        RotateKVStyleAdapter(bits_k=3, bits_v=3),
        CAREKVAdapter(mode="fixed", bits=3, base_quantizer="rotatekv_style",
                       bits_k=3, bits_v=3, max_pages=16),
        # TurboQuant (unsupported)
        TurboQuantStyleUnsupported(bits=3),
        TurboQuantPlusCAREKVUnsupported(bits=3),
    ]


# ─────────────────────────────────────────────
# Per-cell run (mirrors eval_official_kivi_comparison's run_one)
# ─────────────────────────────────────────────

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

    t0 = time.perf_counter()
    reset_peak_gpu()
    try:
        m = adapter.setup_model(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id or 0
        if dataset == "wikitext":
            ppl, n_tok = eval_ppl_wikitext(m, tok, seq_len, num_samples)
        else:
            ppl, n_tok = eval_ppl_synthetic(m, tok, seq_len)
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
    args = ap.parse_args()

    rows: List[ResultRow] = []
    for a in build_adapters():
        r = run_one(a, MODEL_ID, args.dataset, args.seq_len, args.num_samples)
        rows.append(r)
        marker = "✗" if r.official_or_reimpl == "unsupported" else "✓"
        print(f"[BQ-EXP] {marker} {a.name:50s} PPL={r.ppl:9.4f}  "
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
