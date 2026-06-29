"""
tools/eval_route_policies.py
-----------------------------
P1+P2+P4 validation:
  - Iterate CAREKV_ROUTE_POLICY ∈ {joint, separate, k_first, adaptive} × kind ∈ {v, k, both}
  - For each, run TinyLlama PPL with carekv_stored INT3, SEQ_LEN=128.
  - Use absolute budgets (P2) so STORE/READ counts don't collapse.
  - Compare CAREKV_CORRECTION_IMPL=python vs cached: PPL must be bit-equal,
    cached must be faster (P4).
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


def _ppl_run(*, kind: str, route_policy: str, correction_impl: str,
             store_abs_k: int, store_abs_v: int,
             read_abs_k: int, read_abs_v: int,
             seq_len: int, score_normalize: bool):
    reset_debug_stats()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
    text = "KV cache quantization reduces memory usage during decoding. " * 30
    input_ids = tok(text, return_tensors="pt", truncation=True,
                    max_length=seq_len)["input_ids"].to(DEVICE)

    os.environ["CAREKV_PREFILL_MODE"]            = "carekv_stored"
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"]   = kind
    os.environ["CAREKV_K_CORRECTION_SCALE"]      = "0.05"
    os.environ["CAREKV_SCORE_NORMALIZE"]         = "1" if score_normalize else "0"
    os.environ["CAREKV_ROUTE_POLICY"]            = route_policy
    os.environ["CAREKV_CORRECTION_IMPL"]         = correction_impl
    os.environ["CAREKV_STORE_BUDGET_MODE"]       = "absolute"
    os.environ["CAREKV_READ_BUDGET_MODE"]        = "absolute"
    os.environ["CAREKV_STORE_ABS_K"]             = str(store_abs_k)
    os.environ["CAREKV_STORE_ABS_V"]             = str(store_abs_v)
    os.environ["CAREKV_READ_ABS_K"]              = str(read_abs_k)
    os.environ["CAREKV_READ_ABS_V"]              = str(read_abs_v)
    os.environ["CAREKV_DEBUG_STATS"]             = "1"

    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE)
    m.config.use_cache = False
    cfg = m.config; hd = cfg.hidden_size // cfg.num_attention_heads
    cc = CacheConfig(
        num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=3,
        group_size=32, k_channel_group=32, page_size=16, max_pages=128,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        packed_base=True,
        route_policy=route_policy, correction_impl=correction_impl,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=store_abs_k, store_abs_v=store_abs_v,
        read_abs_k=read_abs_k, read_abs_v=read_abs_v,
    )
    m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(input_ids=input_ids, labels=input_ids, use_cache=False)
    dt = time.perf_counter() - t0
    ppl = math.exp(out.loss.item())
    st = get_debug_stats()
    del m; torch.cuda.empty_cache()
    return ppl, dt, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = []
    # Fixed absolute budgets that actually trigger slot reads:
    SAK, SAV, RAK, RAV = 4, 4, 2, 2

    # Matrix: 4 policies × 3 kinds × 2 impls = 24 cells
    # python path is the correctness anchor; cached path must match its PPL.
    for kind in ["v", "k", "both"]:
        for policy in ["separate", "joint", "k_first", "adaptive"]:
            # python first
            ppl_py, t_py, st_py = _ppl_run(
                kind=kind, route_policy=policy, correction_impl="python",
                store_abs_k=SAK, store_abs_v=SAV, read_abs_k=RAK, read_abs_v=RAV,
                seq_len=args.seq_len,
                score_normalize=(policy == "joint"),
            )
            # cached (must equal python PPL bit-by-bit)
            ppl_ca, t_ca, st_ca = _ppl_run(
                kind=kind, route_policy=policy, correction_impl="cached",
                store_abs_k=SAK, store_abs_v=SAV, read_abs_k=RAK, read_abs_v=RAV,
                seq_len=args.seq_len,
                score_normalize=(policy == "joint"),
            )
            diff = abs(ppl_py - ppl_ca)
            speedup = t_py / t_ca if t_ca > 0 else 0
            row = dict(
                kind=kind, route_policy=policy,
                store_abs_k=SAK, store_abs_v=SAV,
                read_abs_k=RAK, read_abs_v=RAV,
                ppl_python=round(ppl_py, 6),
                ppl_cached=round(ppl_ca, 6),
                ppl_diff=round(diff, 8),
                time_python_s=round(t_py, 2),
                time_cached_s=round(t_ca, 2),
                speedup=round(speedup, 2),
                v_reads_py=st_py.get("v_slots_read", 0),
                k_reads_py=st_py.get("k_slots_read", 0),
                v_reads_ca=st_ca.get("v_slots_read", 0),
                k_reads_ca=st_ca.get("k_slots_read", 0),
            )
            rows.append(row)
            print(f"kind={kind:4s} policy={policy:10s}  "
                  f"PPL py={ppl_py:.4f} ca={ppl_ca:.4f} Δ={diff:.2e}  "
                  f"time py={t_py:.1f}s ca={t_ca:.1f}s speedup={speedup:.2f}× "
                  f"V_reads={st_ca.get('v_slots_read',0)} K_reads={st_ca.get('k_slots_read',0)}",
                  flush=True)

    if rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {len(rows)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
