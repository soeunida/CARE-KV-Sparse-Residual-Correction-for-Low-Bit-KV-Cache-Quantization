"""tools/eval_longctx_ppl.py — long-context PPL (PG-19 / WikiText-2) for CARE-KV.

Reviewer ask: move the benchmark to long context. We add **PG-19** PPL at
SL=4096/8192 while **keeping WikiText-2 chunked PPL** (same non-overlapping
windowed protocol) for continuity with the existing paper numbers.

PPL protocol (identical to eval_ppl_dataset.py / baselines.eval_ppl_wikitext):
  concatenate dataset text -> tokenize -> first N*SL tokens -> N non-overlapping
  windows of SL tokens -> per-window model.forward(labels=..., use_cache=False)
  -> PPL = exp(sum(loss * (SL-1)) / total_tokens). CARE-KV per-sequence cache is
  reset between windows.

Modes (KV treatment) reuse the audited baseline adapters:
  fp16, base_quant_int3, carekv_stored_int3 (paper-best, vectorized), turboquant_int3.

Routing-ablation arms (for the query-aware-vs-SL figure) — same paper-best CARE-KV
budget, only the router utility formula changes via CAREKV_BASELINE_SCORE:
  carekv_qaware  = baseline_score "carekv"        (full query-aware utility)
  carekv_random  = baseline_score "random"        (query-agnostic control)
  carekv_magnitude = baseline_score "magnitude_only"

CARE-KV rows print K_reads/V_reads; K_reads+V_reads==0 => router never fired =>
INVALID (CLAUDE.md rule).

NOTE on runtime: CARE-KV uses correction_impl="vectorized" (batched P5 joint+both
path, <=1e-4 vs cached). It is still a prototype and slow at long SL, so N is kept
small and always recorded — read PPL, not wall-clock, as the deliverable.

Run (one mode/GPU):
  CUDA_VISIBLE_DEVICES=1 python tools/eval_longctx_ppl.py \
    --model-id deepseek-ai/deepseek-llm-7b-base --dataset pg19 \
    --seq-lens 4096 8192 --num-samples 8 --modes carekv_qaware \
    --out-csv results/longctx_ppl/deepseek_pg19.csv
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer
from CARE_KV.care_kv.baselines import FP16Adapter, BaseQuantAdapter, CAREKVAdapter
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from CARE_KV.care_kv import get_debug_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# mode -> CAREKV_SELECTOR_VARIANT (routing-ablation knob honored by the
# vectorized correction path; "current" = paper-best query-aware utility).
#   current                   : full query-aware utility (attention x q.r x ...)
#   random                    : query-agnostic random selection (weak control)
#   oracle_residual_magnitude : query-AGNOSTIC, ranks slots by raw residual
#                               magnitude — the fair "informed but query-blind"
#                               baseline that isolates the query-aware benefit.
CAREKV_ARMS = {
    "carekv_stored_int3": "current",
    "carekv_qaware":      "current",
    "carekv_random":      "random",
    "carekv_magnitude":   "oracle_residual_magnitude",
}


def make_adapter(mode, maxp):
    if mode == "fp16":
        return FP16Adapter()
    if mode == "base_quant_int3":
        return BaseQuantAdapter(bits=3)
    if mode == "turboquant_int3":
        return TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=0, use_qjl=True)
    if mode in CAREKV_ARMS:
        return CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                             k_store_mode="post_rope", bits_k=3, bits_v=3,
                             sk=2, sv=4, rk=2, rv=2, max_pages=maxp,
                             correction_impl="vectorized")
    raise ValueError(mode)


def maxp_for(sl):
    return math.ceil(sl / 16) + 16


def load_text(dataset, min_chars):
    from datasets import load_dataset
    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(t for t in ds["text"] if t.strip())
    if dataset == "pg19":
        ds = load_dataset("emozilla/pg19", split="test", streaming=True)
        buf, tot = [], 0
        for ex in ds:
            t = ex.get("text", "")
            if t and t.strip():
                buf.append(t)
                tot += len(t)
            if tot >= min_chars:
                break
        return "\n\n".join(buf)
    raise ValueError(dataset)


def build_windows(tok, text, sl, n):
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    need = sl * n
    if ids.numel() < need:
        raise RuntimeError(f"need {need} tokens, have {ids.numel()}")
    return ids[:need].view(n, sl)


def _reset_carekv_cache(model):
    if not hasattr(model, "modules"):
        return
    for sub in model.modules():
        if hasattr(sub, "reset_cache") and hasattr(sub, "_caches"):
            sub.reset_cache()


@torch.no_grad()
def eval_ppl(model, windows):
    tot_loss, tot_tok = 0.0, 0
    for i in range(windows.shape[0]):
        ids = windows[i:i+1].to(DEVICE)
        _reset_carekv_cache(model)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        n = ids.shape[1] - 1
        tot_loss += float(out.loss.item()) * n
        tot_tok += n
    return math.exp(tot_loss / tot_tok), tot_tok


COLS = ["model_id", "dataset", "seq_len", "num_samples", "mode", "baseline_score",
        "ppl", "total_tokens", "k_reads", "v_reads", "seconds",
        "peak_gpu_mem_MB", "imp_gini", "imp_norm_entropy", "imp_top1pct_mass",
        "status", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--dataset", default="pg19", choices=["pg19", "wikitext"])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192])
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--modes", nargs="+",
                    default=["fp16", "base_quant_int3", "carekv_stored_int3", "turboquant_int3"])
    ap.add_argument("--out-csv", required=True)
    A = ap.parse_args()

    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)
    rows, done = [], set()
    if os.path.exists(A.out_csv) and os.path.getsize(A.out_csv) > 0:
        rows = list(csv.DictReader(open(A.out_csv)))
        done = {(r["model_id"], r["dataset"], r["seq_len"], r["mode"]) for r in rows}

    def flush():
        with open(A.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    tok = AutoTokenizer.from_pretrained(A.model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    max_sl = max(A.seq_lens)
    # ~5 chars/token, big margin for PG-19 windowing
    text = load_text(A.dataset, min_chars=max_sl * A.num_samples * 8 + 200000)
    print(f"[lc] {A.model_id} dataset={A.dataset} SLs={A.seq_lens} N={A.num_samples} "
          f"modes={A.modes}", flush=True)

    for sl in A.seq_lens:
        windows = build_windows(tok, text, sl, A.num_samples)
        maxp = maxp_for(sl)
        for mode in A.modes:
            if (A.model_id, A.dataset, str(sl), mode) in done:
                print(f"[lc] skip {A.dataset} SL{sl} {mode}", flush=True); continue
            bscore = CAREKV_ARMS.get(mode)
            os.environ.pop("CAREKV_SELECTOR_VARIANT", None)
            os.environ.pop("CAREKV_CHUNKED_CORRECTION", None)
            os.environ.pop("CAREKV_CHUNK_SIZE", None)
            if bscore is not None:
                os.environ["CAREKV_SELECTOR_VARIANT"] = bscore
                # The vectorized joint correction materializes a (Q, N, D) tensor
                # (O(S^2) mem) and OOMs at SL>=4096 on 7B. Query-chunking caps it
                # to (chunk, N, D); chunk=256 is ~1GB at SL=8192 and is
                # numerically equivalent to full within <=0.05% PPL (attention.py
                # docstring). Required for the long-context runs.
                os.environ["CAREKV_CHUNKED_CORRECTION"] = "1"
                os.environ["CAREKV_CHUNK_SIZE"] = os.environ.get("LC_CHUNK_SIZE", "256")
            if DEVICE == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            adapter = make_adapter(mode, maxp)
            t0 = time.perf_counter()
            try:
                model = adapter.setup_model(A.model_id)
                ppl, ntok = eval_ppl(model, windows)
                stats = adapter.collect_debug_stats() if hasattr(adapter, "collect_debug_stats") else {}
                kr, vr = int(stats.get("k_reads", 0)), int(stats.get("v_reads", 0))
                peak = (torch.cuda.max_memory_allocated()/1e6) if DEVICE == "cuda" else 0.0
                valid = not (mode in CAREKV_ARMS and kr + vr == 0)
                note = "" if valid else "INVALID: router never fired"
                ds = get_debug_stats()
                ic = ds.get("imp_count", 0) or 0
                imp_gini = round(ds.get("imp_gini_sum", 0.0)/ic, 5) if ic else ""
                imp_ent = round(ds.get("imp_norm_entropy_sum", 0.0)/ic, 5) if ic else ""
                imp_t1 = round(ds.get("imp_top1pct_mass_sum", 0.0)/ic, 5) if ic else ""
                rec = dict(model_id=A.model_id, dataset=A.dataset, seq_len=sl,
                           num_samples=A.num_samples, mode=mode,
                           baseline_score=bscore or "-",
                           ppl=round(ppl, 4), total_tokens=ntok, k_reads=kr, v_reads=vr,
                           seconds=round(time.perf_counter()-t0, 1),
                           peak_gpu_mem_MB=round(peak, 1),
                           imp_gini=imp_gini, imp_norm_entropy=imp_ent,
                           imp_top1pct_mass=imp_t1,
                           status="real" if valid else "invalid", notes=note)
                print(f"[lc] {A.dataset} SL{sl} {mode:20s} PPL={rec['ppl']:.4f} "
                      f"K={kr} V={vr} ({rec['seconds']}s) peak={rec['peak_gpu_mem_MB']}MB"
                      f"{'  <<INVALID' if not valid else ''}", flush=True)
                if hasattr(adapter, "teardown"):
                    try: adapter.teardown()
                    except Exception: pass
                del model
            except Exception as e:
                import traceback; traceback.print_exc()
                rec = dict(model_id=A.model_id, dataset=A.dataset, seq_len=sl,
                           num_samples=A.num_samples, mode=mode, baseline_score=bscore or "-",
                           ppl="", status=f"error:{type(e).__name__}", notes=str(e)[:200])
                if hasattr(adapter, "teardown"):
                    try: adapter.teardown()
                    except Exception: pass
            os.environ.pop("CAREKV_BASELINE_SCORE", None)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            rows.append(rec); flush()

    print(f"[lc] done -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
