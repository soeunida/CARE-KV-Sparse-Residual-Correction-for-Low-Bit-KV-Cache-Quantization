"""
tools/eval_prefill_vectorization.py
------------------------------------
Compare prefill PPL + runtime for the three correction implementations:
  python      — original per-(h, t) loop, no slot cache
  cached      — per-(h, t) loop with pre-unpacked slot cache (P4-cached)
  vectorized  — batched V correction over (h × T) + cached K (P4-vectorized)

Method config locked at the paper-best:
  carekv_stored + both + joint + score_normalize=1 + absolute SK=2 SV=4 RK=2 RV=2
  packed_base=True + scale_quant=int8 + budget=uniform

Reports per cell:
  ppl, prefill_seconds, K_reads, V_reads, peak_gpu_mem_MB.
"""
import argparse, csv, math, os, sys, time
import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats,
)
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _run_one(impl, seq_len):
    os.environ["CAREKV_PREFILL_MODE"]          = "carekv_stored"
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
    os.environ["CAREKV_K_CORRECTION_SCALE"]    = "0.05"
    os.environ["CAREKV_SCORE_NORMALIZE"]       = "1"
    # NB: vectorized V currently doesn't replicate the joint-policy K/V
    # interleaved topk; using `separate` here so all three impls run the
    # same scoring path and the comparison is apples-to-apples.  When
    # joint+both is selected, layer.py auto-falls-back to cached.
    os.environ["CAREKV_ROUTE_POLICY"]          = "separate"
    os.environ["CAREKV_CORRECTION_IMPL"]       = impl
    os.environ["CAREKV_STORE_BUDGET_MODE"]     = "absolute"
    os.environ["CAREKV_READ_BUDGET_MODE"]      = "absolute"
    os.environ["CAREKV_DEBUG_STATS"]           = "1"
    os.environ["CAREKV_BUDGET_POLICY"]         = "uniform"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
    text = "KV cache quantization reduces memory usage during decoding. " * 30
    input_ids = tok(text, return_tensors="pt", truncation=True,
                    max_length=seq_len)["input_ids"].to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    reset_debug_stats()
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
    m.config.use_cache = False
    cfg = m.config; hd = cfg.hidden_size // cfg.num_attention_heads
    cc = CacheConfig(
        num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=3,
        group_size=32, k_channel_group=32, page_size=16, max_pages=128,
        v_token_block=4,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
        packed_base=True, scale_quant="int8",
        route_policy="separate", correction_impl=impl,
        budget_policy="uniform",
    )
    m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(input_ids=input_ids, labels=input_ids, use_cache=False)
    if DEVICE == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ppl = math.exp(out.loss.item())
    st = get_debug_stats()
    peak_mb = (torch.cuda.max_memory_allocated()/1e6) if DEVICE == "cuda" else 0.0
    del m; torch.cuda.empty_cache()
    return dict(impl=impl, ppl=round(ppl, 6), seconds=round(dt, 2),
                K_reads=st.get("k_slots_read", 0),
                V_reads=st.get("v_slots_read", 0),
                peak_gpu_mem_MB=round(peak_mb, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", nargs="+", type=int, default=[64, 128])
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []
    for sl in args.seq_lens:
        cached_t = None
        for impl in ["python", "cached", "vectorized"]:
            try:
                r = _run_one(impl, sl)
                r["seq_len"] = sl
                if impl == "cached": cached_t = r["seconds"]
                if impl == "vectorized" and cached_t:
                    r["speedup_vs_cached"] = round(cached_t / r["seconds"], 2)
                else:
                    r["speedup_vs_cached"] = "-"
                rows.append(r)
                print(f"[prefill-vec] sl={sl} impl={impl:10s}  "
                      f"PPL={r['ppl']:.4f}  {r['seconds']:7.1f}s  "
                      f"K={r['K_reads']} V={r['V_reads']}  "
                      f"peak={r['peak_gpu_mem_MB']:.1f}MB  "
                      f"speedup_vs_cached={r.get('speedup_vs_cached','-')}",
                      flush=True)
            except Exception as e:
                print(f"[prefill-vec] sl={sl} impl={impl}: ERROR {e}", flush=True)
                rows.append(dict(seq_len=sl, impl=impl, ppl=-1, seconds=-1,
                                 K_reads=0, V_reads=0, peak_gpu_mem_MB=-1,
                                 speedup_vs_cached="-", error=str(e)))

    if rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {len(rows)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
