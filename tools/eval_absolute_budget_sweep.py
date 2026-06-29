"""
tools/eval_absolute_budget_sweep.py
------------------------------------
Curated absolute K/V budget sweep at the optimized CARE-KV path:

  CAREKV_PREFILL_MODE=carekv_stored
  CAREKV_PREFILL_RESIDUAL_KIND=both
  CAREKV_ROUTE_POLICY=joint
  CAREKV_SCORE_NORMALIZE=1
  CAREKV_STORE/READ_BUDGET_MODE=absolute
  CAREKV_CORRECTION_IMPL=cached
  CAREKV_PACKED_BASE=1
  CAREKV_SCALE_QUANT=int8
  BASE_BITS=3

Cells include the 7-point user grid plus a (0, 0) invariant row.

Per cell logs:
  PPL, K stored, V stored, K reads, V reads, mean |ΔO_K|, mean |ΔO_V|,
  seconds, total_MB, vs_fp16, route policy, score_normalize, correction_impl.
"""

import argparse, csv, math, os, sys, time
import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats, estimate_memory_bytes,
)
from CARE_KV.care_kv.llama_patch import CAREKVLlamaAttention


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _scan_stored(model):
    """Sum stored K/V slot counts across all CAREKV-patched layers."""
    used_k = used_v = 0
    for m in model.modules():
        if isinstance(m, CAREKVLlamaAttention):
            for cache in m._caches.values():
                k, v = cache.num_stored_residual_slots()
                used_k += k; used_v += v
    return used_k, used_v


def _run_cell(*, store_abs_k, store_abs_v, read_abs_k, read_abs_v,
              policy, score_norm, correction_impl, base_bits, seq_len, kind):
    os.environ["CAREKV_PREFILL_MODE"]          = "carekv_stored"
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = kind
    os.environ["CAREKV_K_CORRECTION_SCALE"]    = "0.05"
    os.environ["CAREKV_SCORE_NORMALIZE"]       = "1" if score_norm else "0"
    os.environ["CAREKV_ROUTE_POLICY"]          = policy
    os.environ["CAREKV_CORRECTION_IMPL"]       = correction_impl
    os.environ["CAREKV_STORE_BUDGET_MODE"]     = "absolute"
    os.environ["CAREKV_READ_BUDGET_MODE"]      = "absolute"
    os.environ["CAREKV_DEBUG_STATS"]           = "1"

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
    text = "KV cache quantization reduces memory usage during decoding. " * 30
    input_ids = tok(text, return_tensors="pt", truncation=True,
                    max_length=seq_len)["input_ids"].to(DEVICE)

    reset_debug_stats()
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
    m.config.use_cache = False
    cfg = m.config; hd = cfg.hidden_size // cfg.num_attention_heads
    cc = CacheConfig(
        num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=base_bits,
        group_size=32, k_channel_group=32, page_size=16, max_pages=128,
        # absolute mode → ratio is effectively unused but kept at 0 for invariant
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=store_abs_k, store_abs_v=store_abs_v,
        read_abs_k=read_abs_k, read_abs_v=read_abs_v,
        packed_base=True, scale_quant="int8",
        route_policy=policy, correction_impl=correction_impl,
        v_token_block=4,
    )
    m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(input_ids=input_ids, labels=input_ids, use_cache=False)
    dt = time.perf_counter() - t0
    ppl = math.exp(out.loss.item())
    stored_k, stored_v = _scan_stored(m)
    st = get_debug_stats()
    nq = max(st.get("n_queries", 1), 1)
    mem = estimate_memory_bytes(cc, seq_len)
    del m; torch.cuda.empty_cache()

    return dict(
        store_abs_k=store_abs_k, store_abs_v=store_abs_v,
        read_abs_k=read_abs_k, read_abs_v=read_abs_v,
        route_policy=policy, score_normalize=score_norm,
        correction_impl=correction_impl, base_bits=base_bits,
        ppl=round(ppl, 6),
        stored_K=stored_k, stored_V=stored_v,
        K_reads=st.get("k_slots_read", 0),
        V_reads=st.get("v_slots_read", 0),
        mean_delta_K=round(st.get("delta_k_norm_sum", 0.0) / nq, 6),
        mean_delta_V=round(st.get("delta_v_norm_sum", 0.0) / nq, 6),
        seconds=round(dt, 2),
        total_MB=round(mem["total_bytes"] / 1e6, 4),
        vs_fp16=round(mem["compression_vs_fp16"], 4),
        current_clean_run=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    # Curated 7-point grid + R=0 invariant row.
    # (sk, sv, rk, rv, label)
    grid = [
        (0, 0, 0, 0, "invariant_zero"),    # acceptance #1: R=0 == base_quant
        (1, 1, 1, 1, "balanced_1"),
        (2, 1, 2, 1, "k_heavy_2_1"),
        (1, 2, 1, 2, "v_heavy_1_2"),
        (2, 2, 2, 2, "balanced_2"),
        (4, 2, 4, 2, "k_heavy_4_2"),
        (2, 4, 2, 4, "v_heavy_2_4"),
        (4, 4, 4, 4, "balanced_4"),
    ]

    rows = []
    for sk, sv, rk, rv, label in grid:
        try:
            row = _run_cell(
                store_abs_k=sk, store_abs_v=sv,
                read_abs_k=rk, read_abs_v=rv,
                policy="joint", score_norm=True,
                correction_impl="cached", base_bits=3,
                seq_len=args.seq_len, kind="both",
            )
            row["label"] = label
            rows.append(row)
            print(f"[abs-sweep] {label:18s} SK={sk} SV={sv} RK={rk} RV={rv}  "
                  f"PPL={row['ppl']:.4f}  K_reads={row['K_reads']:6d} V_reads={row['V_reads']:6d}  "
                  f"mem={row['total_MB']:.2f}MB ({row['vs_fp16']:.3f}× fp16)  "
                  f"({row['seconds']:.1f}s)", flush=True)
        except Exception as e:
            print(f"[abs-sweep] {label}: ERROR {type(e).__name__}: {e}", flush=True)
            rows.append(dict(label=label, store_abs_k=sk, store_abs_v=sv,
                             read_abs_k=rk, read_abs_v=rv, ppl=-1,
                             error=str(e), current_clean_run=True))

    if rows:
        # Use the keys from a successful row for the header
        keys = None
        for r in rows:
            if r.get("ppl", -1) >= 0:
                keys = list(r.keys()); break
        if keys is None: keys = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {len(rows)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
