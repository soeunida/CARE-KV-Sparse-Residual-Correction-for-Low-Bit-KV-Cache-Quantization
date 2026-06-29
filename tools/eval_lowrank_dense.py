"""tools/eval_lowrank_dense.py — Phase 0 (eval-mode) for low-rank dense correction.

WikiText-2 PPL of INT3 base + rank-r dense correction (upper bound: exact
residual-SVD subspace + fp coefficients), gated in layer.py to base_quant mode
via CAREKV_LOWRANK_RANK. Tests whether modelling the residual's low-rank
outlier-channel structure beats CARE-KV's sparse routing — WITHOUT rotation.

Reference (TinyLlama, INT3): base_quant N=4 16.20 / N=16 22.58;
uniform+CARE-KV N=4 13.46 / N=16 17.885 (bar to beat).
Outputs: <out-csv> per-rank PPL (rank=0 == base_quant).
"""
from __future__ import annotations
import argparse, csv, os, sys, time

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import CacheConfig, patch_llama_model, reset_all_caches
from CARE_KV.care_kv.cache import apply_carekv_env_overrides
from CARE_KV.care_kv.baselines.common import eval_ppl_wikitext, DEVICE

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def build(rank: int, kind: str, bits: int, head_dim_hint: int, max_pages: int):
    env = dict(
        CAREKV_PREFILL_MODE="base_quant",
        CAREKV_BASE_BITS=str(bits),
        CAREKV_PACKED_BASE="1",
        CAREKV_SCALE_QUANT="int8",
        CAREKV_GROUP_SIZE="32",
        CAREKV_STORE_BUDGET_MODE="absolute", CAREKV_READ_BUDGET_MODE="absolute",
        CAREKV_STORE_ABS_K="0", CAREKV_STORE_ABS_V="0",
        CAREKV_READ_ABS_K="0", CAREKV_READ_ABS_V="0",
        CAREKV_LOWRANK_RANK=str(rank),
        CAREKV_LOWRANK_KIND=kind,
        CAREKV_DEBUG_STATS="1",
    )
    for k, v in env.items():
        os.environ[k] = v
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None)
    m.config.use_cache = False
    cfg = m.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    kw = dict(num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
              num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=bits,
              group_size=32, k_channel_group=32, page_size=16, max_pages=max_pages,
              v_token_block=4, sketch_dim=16,
              store_budget_ratio=0.0, read_budget_ratio=0.0,
              store_budget_mode="absolute", read_budget_mode="absolute")
    apply_carekv_env_overrides(kw)
    m = patch_llama_model(m, CacheConfig(**kw))
    reset_all_caches(m); m.eval()
    return m


def main():
    global MODEL_ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--kind", default="both", choices=["k", "v", "both"])
    ap.add_argument("--ranks", default="0,1,2,4,8")
    args = ap.parse_args()
    MODEL_ID = args.model_id
    ranks = [int(x) for x in args.ranks.split(",") if x.strip() != ""]
    max_pages = max(16, args.seq_len // 16 + 4)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    rows = []
    for r in ranks:
        t0 = time.perf_counter()
        try:
            m = build(r, args.kind, args.bits, 0, max_pages)
            ppl, ntok = eval_ppl_wikitext(m, tok, args.seq_len, args.num_samples)
            dt = time.perf_counter() - t0
            row = dict(rank=r, kind=(args.kind if r > 0 else "-"),
                       bits=args.bits, ppl=round(float(ppl), 4),
                       evaluated_tokens=int(ntok), seconds=round(dt, 1),
                       seq_len=args.seq_len, num_samples=args.num_samples,
                       model_id=MODEL_ID)
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            row = dict(rank=r, ppl=0.0, error=f"{type(e).__name__}: {e}")
        rows.append(row)
        print(f"[LOWRANK] rank={r:<2d} kind={args.kind:5s} "
              f"PPL={row.get('ppl',0):9.4f}  ({row.get('seconds','?')}s)", flush=True)

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
