"""
tools/eval_layer_budget_policies.py
------------------------------------
Phase E: compare per-layer budget allocation policies at the optimized
CARE-KV path with fixed total budget (SK=2, SV=4, RK=2, RV=2 abs).

Policies:
  uniform     — every layer gets the global budget (mean=1.0 multiplier).
  u_shaped    — built-in 0.5+1.5·|2l/(L−1)−1| profile; edge layers get ~1.56×,
                middle ~0.44×; mean preserved → total reads ~ unchanged.
  sensitivity — uses cfg.layer_sensitivity directly (defaults to all 1.0 →
                same as uniform).  For this eval we set sensitivity to a
                literature-inspired U-shape sharper than the built-in
                u_shaped formula, so the two adaptive policies are
                distinguishable.

For each policy:
  - run TinyLlama prefill PPL
  - log K_reads, V_reads (totals)
  - log per-layer stored K/V slot counts and per-layer read K/V counts
    (collected by walking CAREKVLlamaAttention._caches[0] for each layer)
"""

import argparse, csv, json, math, os, sys, time
import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats, estimate_memory_bytes,
    layer_budget_multiplier,
)
from CARE_KV.care_kv.llama_patch import CAREKVLlamaAttention

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _per_layer_stored(model):
    """Walk each CAREKVLlamaAttention wrapper and report its seq-0 cache's
    stored slot count, keyed by layer_id."""
    out = {}
    for m in model.modules():
        if isinstance(m, CAREKVLlamaAttention) and 0 in m._caches:
            k, v = m._caches[0].num_stored_residual_slots()
            out[m.layer_id] = {"stored_K": k, "stored_V": v}
    return out


def _run_one(policy, layer_sensitivity, seq_len):
    os.environ["CAREKV_PREFILL_MODE"]          = "carekv_stored"
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
    os.environ["CAREKV_K_CORRECTION_SCALE"]    = "0.05"
    os.environ["CAREKV_SCORE_NORMALIZE"]       = "1"
    os.environ["CAREKV_ROUTE_POLICY"]          = "joint"
    os.environ["CAREKV_CORRECTION_IMPL"]       = "cached"
    os.environ["CAREKV_STORE_BUDGET_MODE"]     = "absolute"
    os.environ["CAREKV_READ_BUDGET_MODE"]      = "absolute"
    os.environ["CAREKV_DEBUG_STATS"]           = "1"
    os.environ["CAREKV_BUDGET_POLICY"]         = policy
    if layer_sensitivity is not None:
        os.environ["CAREKV_LAYER_SENSITIVITY"] = ",".join(str(x) for x in layer_sensitivity)
    else:
        os.environ.pop("CAREKV_LAYER_SENSITIVITY", None)

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
        num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=3,
        group_size=32, k_channel_group=32, page_size=16, max_pages=128,
        v_token_block=4,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
        packed_base=True, scale_quant="int8",
        route_policy="joint", correction_impl="cached",
        budget_policy=policy,
        layer_sensitivity=layer_sensitivity,
    )
    m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()

    # Per-layer multiplier preview (independent of model)
    multipliers = [layer_budget_multiplier(cc, l) for l in range(cc.num_layers)]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(input_ids=input_ids, labels=input_ids, use_cache=False)
    dt = time.perf_counter() - t0
    ppl = math.exp(out.loss.item())

    per_layer = _per_layer_stored(m)
    st = get_debug_stats()
    del m; torch.cuda.empty_cache()

    return {
        "policy": policy,
        "ppl": round(ppl, 6),
        "K_reads_total": st.get("k_slots_read", 0),
        "V_reads_total": st.get("v_slots_read", 0),
        "seconds": round(dt, 2),
        "multipliers": multipliers,
        "per_layer_stored": per_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", default=None,
                    help="per-layer breakdown JSON (default: replace .csv with .json)")
    args = ap.parse_args()

    if args.out_json is None:
        args.out_json = args.out_csv.replace(".csv", "_per_layer.json")

    # Cells:
    #  uniform                         — default profile
    #  u_shaped                        — built-in U formula (edge ~1.56, mid ~0.44)
    #  sensitivity, sharp U            — manually-supplied stronger U
    #  sensitivity, default 1.0 (≡ uniform) — sanity row
    L = 22
    # "Sharp U" — 2.0 at top 4 + bottom 4 layers, 0.5 in middle 14
    sharp_u = [2.0]*4 + [0.5]*14 + [2.0]*4

    rows = []
    for label, policy, sens in [
        ("uniform_baseline",          "uniform",     None),
        ("u_shaped_builtin",          "u_shaped",    None),
        ("sensitivity_sharp_u",       "sensitivity", sharp_u),
        ("sensitivity_default_uni",   "sensitivity", None),
    ]:
        r = _run_one(policy, sens, args.seq_len)
        r["label"] = label
        rows.append(r)
        print(f"[layer-budget] {label:28s} policy={policy:11s} "
              f"PPL={r['ppl']:.4f}  K_reads={r['K_reads_total']:6d} "
              f"V_reads={r['V_reads_total']:6d}  ({r['seconds']:.1f}s)",
              flush=True)

    csv_cols = ["label","policy","ppl","K_reads_total","V_reads_total","seconds"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {args.out_csv}")

    # Per-layer breakdown JSON
    with open(args.out_json, "w") as f:
        json.dump({r["label"]: {
            "policy": r["policy"],
            "ppl": r["ppl"],
            "multipliers": r["multipliers"],
            "per_layer_stored": r["per_layer_stored"],
        } for r in rows}, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
