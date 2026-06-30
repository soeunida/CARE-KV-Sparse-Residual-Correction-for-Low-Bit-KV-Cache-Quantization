"""tools/diag_router_bottleneck.py — Phase-0 router-bottleneck diagnostic.

Two questions, one cheap run (TinyLlama, WikiText-2 PPL):

  (1) Is the STORED-SLOT SELECTION the bottleneck (router) or the residual
      REPRESENTATION (INT3 base + 4-bit residual)?
        - carekv_eval (CAREKV_PREFILL_RESIDUAL_RATIO=1.0) corrects with ALL
          residuals → upper bound of the representation.
        - carekv_stored = real slot-faithful method (SK2SV4 RK2RV2).
        gap(stored, eval) large  ⇒ selection/budget is the bottleneck → fix router.
        eval still far from fp16  ⇒ representation is the limit → not just router.

  (2) Does K-router scoring quality matter? carekv_stored with sketch_dim 16/32/64.

Outputs: <out-csv>.
"""
from __future__ import annotations
import argparse, csv, os, sys, time

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import CacheConfig, patch_llama_model, reset_all_caches, get_debug_stats, reset_debug_stats
from CARE_KV.care_kv.cache import apply_carekv_env_overrides
from CARE_KV.care_kv.baselines.common import eval_ppl_wikitext, DEVICE

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build(prefill_mode, sketch_dim=16, sk=2, sv=4, rk=2, rv=2, bits=3,
          eval_full=False, max_pages=16, corr_impl="vectorized"):
    env = dict(
        CAREKV_PREFILL_MODE=prefill_mode,
        CAREKV_PREFILL_RESIDUAL_KIND="both",
        CAREKV_ROUTE_POLICY="joint",
        CAREKV_SCORE_NORMALIZE="1",
        CAREKV_CORRECTION_IMPL=corr_impl,
        CAREKV_BUDGET_POLICY="uniform",
        CAREKV_PACKED_BASE="1",
        CAREKV_SCALE_QUANT="int8",
        CAREKV_BASE_BITS=str(bits),
        CAREKV_GROUP_SIZE="32",
        CAREKV_SKETCH_DIM=str(sketch_dim),
        CAREKV_STORE_BUDGET_MODE="absolute", CAREKV_READ_BUDGET_MODE="absolute",
        CAREKV_STORE_ABS_K=str(sk), CAREKV_STORE_ABS_V=str(sv),
        CAREKV_READ_ABS_K=str(rk), CAREKV_READ_ABS_V=str(rv),
        CAREKV_DEBUG_STATS="1",
    )
    if eval_full:   # carekv_eval upper bound: correct with ALL residuals
        env["CAREKV_PREFILL_RESIDUAL_RATIO"] = "1.0"
        env["CAREKV_PREFILL_MIN_RESIDUALS"] = "9999"
        env["CAREKV_K_CORRECTION_SCALE"] = "0.1"
    for k, v in env.items():
        os.environ[k] = v
    for k in ("CAREKV_PREFILL_RESIDUAL_RATIO", "CAREKV_PREFILL_MIN_RESIDUALS"):
        if not eval_full and k in os.environ:
            del os.environ[k]
    reset_debug_stats()
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16,
                                         device_map=DEVICE if DEVICE == "cuda" else None)
    m.config.use_cache = False
    cfg = m.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    kw = dict(num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
              num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=bits,
              group_size=32, k_channel_group=32, page_size=16, max_pages=max_pages,
              v_token_block=4, sketch_dim=sketch_dim,
              store_budget_ratio=0.0, read_budget_ratio=0.0,
              store_budget_mode="absolute", read_budget_mode="absolute")
    apply_carekv_env_overrides(kw)
    m = patch_llama_model(m, CacheConfig(**kw))
    reset_all_caches(m); m.eval()
    return m


ARMS = [
    # label,              prefill_mode,    sketch, eval_full, corr_impl
    ("fp16",              "fp",            16, False, "vectorized"),
    ("base_int3",         "base_quant",    16, False, "vectorized"),
    ("carekv_vec_sk16",   "carekv_stored", 16, False, "vectorized"),  # real method, fast path
    ("carekv_vec_sk32",   "carekv_stored", 32, False, "vectorized"),
    ("carekv_vec_sk64",   "carekv_stored", 64, False, "vectorized"),
    ("carekv_cached_sk16","carekv_stored", 16, False, "cached"),      # validation: must ≈ vec_sk16 (slow)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=2)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    mp = max(16, args.seq_len // 16 + 4)

    rows = []
    for label, mode, sk_dim, full, corr in ARMS:
        t0 = time.perf_counter()
        try:
            m = build(mode, sketch_dim=sk_dim, eval_full=full, max_pages=mp, corr_impl=corr)
            ppl, ntok = eval_ppl_wikitext(m, tok, args.seq_len, args.num_samples)
            st = get_debug_stats()
            row = dict(arm=label, prefill_mode=mode, sketch_dim=sk_dim, corr_impl=corr,
                       ppl=round(float(ppl), 4), k_reads=int(st.get("k_slots_read", 0)),
                       v_reads=int(st.get("v_slots_read", 0)),
                       evaluated_tokens=int(ntok), seconds=round(time.perf_counter()-t0, 1))
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            row = dict(arm=label, ppl=0.0, error=f"{type(e).__name__}: {e}")
        rows.append(row)
        print(f"[DIAG] {label:20s} PPL={row.get('ppl',0):9.4f} "
              f"K={row.get('k_reads','?')} V={row.get('v_reads','?')} "
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
