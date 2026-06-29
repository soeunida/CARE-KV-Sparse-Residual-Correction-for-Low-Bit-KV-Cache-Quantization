"""
tools/debug_stored_slot_reads.py
---------------------------------
Targeted reproducer for the carekv_stored zero-reads issue.

For each test case (A-E) runs one TinyLlama forward at SEQ_LEN, then prints:
    PPL                       — perplexity over the sequence
    store_budget_ratio        — read back from the live CacheConfig
    read_budget_ratio
    stored V/K slot counts    — across all caches in the model
    routing candidate counts  — average across queries
    average topk per query    — actually selected slots
    read V/K slot counts      — from CAREKV_DEBUG_STATS
    correction applied        — boolean (true iff ΔO != 0)
    mean |ΔO_V|, mean |ΔO_K|
"""

from __future__ import annotations
import argparse, math, os, sys, time
import torch
sys.path.insert(0, "/home/soeun")
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats,
)
from CARE_KV.care_kv.llama_patch import CAREKVLlamaAttention

MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_tokenized(seq_len):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None or tok.pad_token_id < 0:
        tok.pad_token_id = tok.eos_token_id or 0
    text = ("KV cache quantization reduces memory usage during decoding. " * 30)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)
    return tok, enc["input_ids"].to(DEVICE)


def _scan_caches(model):
    """Walk all CAREKVLlamaAttention wrappers and sum stored slot counts."""
    used_k = used_v = 0
    n_caches = 0
    cfg_seen = None
    for m in model.modules():
        if isinstance(m, CAREKVLlamaAttention):
            for cache in m._caches.values():
                n_caches += 1
                k, v = cache.num_stored_residual_slots()
                used_k += k; used_v += v
                cfg_seen = cache.cfg
    return n_caches, used_k, used_v, cfg_seen


def _make_model(care_cfg=None):
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    model.config.use_cache = False
    if care_cfg is not None:
        model = patch_llama_model(model, care_cfg)
        reset_all_caches(model)
    model.eval()
    return model


def _care_cfg(model, *, base_bits, store_b, read_b, kind, k_scale):
    c = model.config
    hd = c.hidden_size // c.num_attention_heads
    return CacheConfig(
        num_layers=c.num_hidden_layers,
        num_heads=c.num_attention_heads,
        num_kv_heads=c.num_key_value_heads,
        head_dim=hd,
        page_size=16,
        max_pages=64,
        base_bits=base_bits, group_size=32, k_channel_group=32, v_token_block=4,
        store_budget_ratio=store_b, read_budget_ratio=read_b,
        sketch_dim=16, packed_base=True,
    )


def _run_one(label, *, base_bits, store_b, read_b, prefill_mode, kind, k_scale,
             input_ids):
    print(f"\n===== {label} =====", flush=True)
    print(f"  prefill_mode={prefill_mode}  base_bits={base_bits}  "
          f"store_b={store_b}  read_b={read_b}  kind={kind}  k_scale={k_scale}")

    reset_debug_stats()
    os.environ["CAREKV_PREFILL_MODE"] = prefill_mode
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = kind
    os.environ["CAREKV_K_CORRECTION_SCALE"] = str(k_scale)
    os.environ["CAREKV_DEBUG_STATS"] = "1"

    model = _make_model()
    if prefill_mode in {"base_quant", "carekv_eval", "carekv_stored"}:
        cc = _care_cfg(model, base_bits=base_bits, store_b=store_b, read_b=read_b,
                       kind=kind, k_scale=k_scale)
        model = patch_llama_model(model, cc); reset_all_caches(model)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    dt = time.perf_counter() - t0
    ppl = math.exp(out.loss.item())

    n_caches, stored_k, stored_v, cfg_seen = _scan_caches(model)
    st = get_debug_stats()
    nq = max(st.get("n_queries", 1), 1)

    print(f"  PPL = {ppl:.4f}   ({dt:.1f} s)")
    if cfg_seen is not None:
        print(f"  live cfg: store_budget_ratio={cfg_seen.store_budget_ratio}  "
              f"read_budget_ratio={cfg_seen.read_budget_ratio}")
    print(f"  caches seen: {n_caches}")
    print(f"  stored slots: V={stored_v}  K={stored_k}  (across all layers, this seq)")
    print(f"  reads      : V={st.get('v_slots_read',0)}  K={st.get('k_slots_read',0)}  over {nq} queries")
    if nq > 0:
        avg_v = st.get('v_slots_read',0) / nq
        avg_k = st.get('k_slots_read',0) / nq
        print(f"  avg per query: V={avg_v:.3f}  K={avg_k:.3f}  (= effective topk)")
    delta_v_norm = st.get('delta_v_norm_sum',0.0)/nq
    delta_k_norm = st.get('delta_k_norm_sum',0.0)/nq
    o_base_norm  = st.get('o_base_norm_sum',0.0)/nq
    print(f"  ⟨|ΔO_V|⟩ = {delta_v_norm:.4e}   ⟨|ΔO_K|⟩ = {delta_k_norm:.4e}   ⟨|O_base|⟩ = {o_base_norm:.4e}")
    correction_applied = (delta_v_norm + delta_k_norm) > 0
    print(f"  correction_applied = {correction_applied}")

    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return dict(label=label, ppl=ppl, prefill_mode=prefill_mode,
                base_bits=base_bits, store_b=store_b, read_b=read_b,
                kind=kind, k_scale=k_scale,
                stored_k=stored_k, stored_v=stored_v,
                read_k=st.get('k_slots_read',0), read_v=st.get('v_slots_read',0),
                n_queries=nq,
                delta_v_norm=delta_v_norm, delta_k_norm=delta_k_norm,
                o_base_norm=o_base_norm,
                correction_applied=correction_applied,
                seconds=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _, input_ids = _build_tokenized(args.seq_len)

    rows = []
    rows.append(_run_one("A. base_quant INT3 (baseline)",
                         base_bits=3, store_b=0.10, read_b=0.0,
                         prefill_mode="base_quant", kind="both", k_scale=0.1,
                         input_ids=input_ids))
    rows.append(_run_one("B. carekv_stored V-only S=0.10 R=0.03",
                         base_bits=3, store_b=0.10, read_b=0.03,
                         prefill_mode="carekv_stored", kind="v", k_scale=0.1,
                         input_ids=input_ids))
    rows.append(_run_one("C. carekv_stored V-only S=0.10 R=0.10",
                         base_bits=3, store_b=0.10, read_b=0.10,
                         prefill_mode="carekv_stored", kind="v", k_scale=0.1,
                         input_ids=input_ids))
    rows.append(_run_one("D. carekv_stored both    S=0.50 R=0.30",
                         base_bits=3, store_b=0.50, read_b=0.30,
                         prefill_mode="carekv_stored", kind="both", k_scale=0.1,
                         input_ids=input_ids))
    rows.append(_run_one("E. carekv_stored V-only S=0.10 R=0.00 (must match A)",
                         base_bits=3, store_b=0.10, read_b=0.0,
                         prefill_mode="carekv_stored", kind="v", k_scale=0.1,
                         input_ids=input_ids))

    # Acceptance assertions
    print("\n=== ACCEPTANCE CHECKS ===")
    A = rows[0]; B = rows[1]; C = rows[2]; D = rows[3]; E = rows[4]
    def cmp_assert(label, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    cmp_assert("E (R=0) ≡ A (base_quant) exactly", abs(E["ppl"]-A["ppl"]) < 1e-4,
               f"|diff|={abs(E['ppl']-A['ppl']):.2e}")
    cmp_assert("B (V-only, R=0.03) reads V > 0", B["read_v"] > 0,
               f"V_reads={B['read_v']}")
    cmp_assert("C (V-only, R=0.10) reads V > 0", C["read_v"] > 0,
               f"V_reads={C['read_v']}")
    cmp_assert("D (both, R=0.30) reads V>0 OR K>0", D["read_v"] + D["read_k"] > 0,
               f"V={D['read_v']} K={D['read_k']}")
    cmp_assert("D PPL differs from A (correction did something)",
               abs(D["ppl"] - A["ppl"]) > 1e-4,
               f"A={A['ppl']:.4f} D={D['ppl']:.4f}")

    with open(args.out, "w") as f:
        f.write("=== carekv_stored zero-reads diagnosis ===\n")
        f.write(f"SEQ_LEN={args.seq_len}   MODEL={MODEL_ID}\n\n")
        for r in rows:
            f.write(f"\n--- {r['label']} ---\n")
            for k, v in r.items():
                if k == "label": continue
                f.write(f"  {k:18s} = {v}\n")
        f.write("\n=== ACCEPTANCE ===\n")
        f.write(f"  E (R=0) == A             : {'PASS' if abs(E['ppl']-A['ppl']) < 1e-4 else 'FAIL'}\n")
        f.write(f"  B V_reads > 0            : {'PASS' if B['read_v']>0 else 'FAIL'}\n")
        f.write(f"  C V_reads > 0            : {'PASS' if C['read_v']>0 else 'FAIL'}\n")
        f.write(f"  D V+K reads > 0          : {'PASS' if D['read_v']+D['read_k']>0 else 'FAIL'}\n")
        f.write(f"  D PPL differs from A     : {'PASS' if abs(D['ppl']-A['ppl'])>1e-4 else 'FAIL'}\n")


if __name__ == "__main__":
    main()
