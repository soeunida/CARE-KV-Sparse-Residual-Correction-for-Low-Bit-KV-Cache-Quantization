"""
tools/bench_latency.py
-----------------------
Decode-latency benchmark for CARE-KV.

Measures, per (mode, base_bits, prompt_len) cell:
    prefill_ms              — wall-clock of the first forward (T input tokens)
    decode_ms_per_token     — mean over `new_tokens` decode steps
    tokens_per_sec          — new_tokens / total_decode_time
    peak_gpu_mem_MB         — torch.cuda.max_memory_allocated()

All modes use use_cache=True (Phase G validated).  carekv_stored's per-token
latency currently includes a full re-prefill (Phase G v1 limitation) — that
cost is honest and shows up here as a large decode_ms_per_token.

Emits one CSV row per measurement with the columns the user spec requires:
    model, mode, base_bits, prefill_mode, store_budget, read_budget,
    prompt_len, new_tokens, prefill_ms, decode_ms_per_token,
    tokens_per_sec, peak_gpu_mem_MB
"""

from __future__ import annotations
import argparse, csv, gc, os, sys, time
from typing import Any, Dict, List, Optional
import torch

sys.path.insert(0, "/home/soeun")
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats,
)

MODEL_ID_DEFAULT = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_prompt(tok, prompt_len: int) -> torch.Tensor:
    base = ("KV cache quantization reduces memory usage during autoregressive "
            "decoding of large language models. " * 50)
    enc = tok(base, return_tensors="pt", truncation=True, max_length=prompt_len)
    return enc["input_ids"].to(DEVICE)


def _make_model(model_id: str, care_cfg: Optional[CacheConfig] = None):
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    m.config.use_cache = True
    m.generation_config.use_cache = True
    if care_cfg is not None:
        m = patch_llama_model(m, care_cfg)
        reset_all_caches(m)
    m.eval()
    return m


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _measure(model, tok, prompt_len: int, new_tokens: int) -> Dict[str, float]:
    """Run one prefill + N decode steps; return prefill_ms, decode_ms_per_token,
    tokens_per_sec, peak_gpu_mem_MB."""
    input_ids = _build_prompt(tok, prompt_len)
    pad = tok.pad_token_id

    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warm one short generation so first-time autograd / cudnn init isn't
    # billed to the prefill timing.
    with torch.no_grad():
        _ = model.generate(input_ids=input_ids[:, :min(8, prompt_len)],
                           max_new_tokens=2, do_sample=False,
                           use_cache=True, pad_token_id=pad)
    if hasattr(model, "modules"):
        for m in model.modules():
            if hasattr(m, "reset_cache"):
                m.reset_cache()
    _sync()

    # ── Prefill timing: forward only on the prompt, return cache so we
    # can hand it back into a second forward for the decode loop. ──────
    _sync(); t0 = time.perf_counter()
    with torch.no_grad():
        out_prefill = model(input_ids=input_ids, use_cache=True)
    _sync()
    prefill_s = time.perf_counter() - t0
    past = out_prefill.past_key_values

    # ── Decode loop: feed one token at a time using past_key_values ────
    # Use the prompt's last token as the seed; subsequent tokens come from
    # argmax of the previous step's logits.
    next_tok = input_ids[:, -1:].clone()
    _sync(); t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(new_tokens):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    _sync()
    decode_s = time.perf_counter() - t0

    peak_mb = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else 0.0
    return {
        "prefill_ms": prefill_s * 1000,
        "decode_ms_per_token": (decode_s / new_tokens) * 1000,
        "tokens_per_sec": new_tokens / decode_s,
        "peak_gpu_mem_MB": peak_mb,
    }


def _care_cfg(model, *, base_bits: int, store_b: float, read_b: float,
              packed_base: bool = True,
              optimized: bool = False) -> CacheConfig:
    """`optimized=True` switches to the paper-best absolute-budget config
    (SK=2 SV=4 RK=2 RV=2, joint policy + score_normalize + cached + int8 scales)
    instead of the legacy ratio knobs."""
    c = model.config
    hd = c.hidden_size // c.num_attention_heads
    kw = dict(
        num_layers=c.num_hidden_layers,
        num_heads=c.num_attention_heads,
        num_kv_heads=c.num_key_value_heads,
        head_dim=hd,
        page_size=16, max_pages=256,
        base_bits=base_bits, group_size=32, k_channel_group=32, v_token_block=4,
        sketch_dim=16, packed_base=packed_base,
    )
    if optimized:
        kw.update(
            store_budget_mode="absolute", read_budget_mode="absolute",
            store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
            store_budget_ratio=0.0, read_budget_ratio=0.0,
            route_policy="joint", correction_impl="cached",
            budget_policy="uniform", scale_quant="int8",
        )
    else:
        kw.update(store_budget_ratio=store_b, read_budget_ratio=read_b)
    return CacheConfig(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    ap.add_argument("--prompt-lens", nargs="+", type=int, default=[128, 512, 1024])
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--modes", nargs="+",
                    default=["fp16", "base_quant_int3", "carekv_stored_int3",
                             "base_quant_int2", "carekv_stored_int2"])
    ap.add_argument("--carekv-new-tokens", type=int, default=8,
                    help="reduced new-token count for carekv_stored modes "
                         "(re-prefill cost makes 64 impractical without "
                         "incremental decode; default 8)")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None or tok.pad_token_id < 0:
        tok.pad_token_id = tok.eos_token_id or 0

    rows: List[Dict[str, Any]] = []
    print(f"# bench_latency.py  model={args.model_id}  new_tokens={args.new_tokens} "
          f"(carekv={args.carekv_new_tokens})  prompt_lens={args.prompt_lens}",
          flush=True)

    for prompt_len in args.prompt_lens:
        for mode_label in args.modes:
            try:
                is_carekv_stored = mode_label.startswith("carekv_stored")
                effective_new = args.carekv_new_tokens if is_carekv_stored else args.new_tokens

                if mode_label == "fp16":
                    model = _make_model(args.model_id, care_cfg=None)
                    bits = 16; prefill_mode = "fp"; sb = rb = 0.0
                elif mode_label == "base_quant_int3":
                    bits, sb, rb = 3, 0.10, 0.0
                    base = _make_model(args.model_id, care_cfg=None)
                    cc = _care_cfg(base, base_bits=bits, store_b=sb, read_b=rb)
                    os.environ["CAREKV_PREFILL_MODE"] = "base_quant"
                    model = patch_llama_model(base, cc); reset_all_caches(model)
                    prefill_mode = "base_quant"
                elif mode_label == "carekv_stored_int3":
                    bits, sb, rb = 3, 0.0, 0.0
                    base = _make_model(args.model_id, care_cfg=None)
                    cc = _care_cfg(base, base_bits=bits, store_b=sb, read_b=rb,
                                   optimized=True)
                    os.environ["CAREKV_PREFILL_MODE"]          = "carekv_stored"
                    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
                    os.environ["CAREKV_ROUTE_POLICY"]          = "joint"
                    os.environ["CAREKV_SCORE_NORMALIZE"]       = "1"
                    os.environ["CAREKV_CORRECTION_IMPL"]       = "cached"
                    os.environ["CAREKV_DEBUG_STATS"]           = "1"
                    model = patch_llama_model(base, cc); reset_all_caches(model)
                    prefill_mode = "carekv_stored_optimized"
                elif mode_label == "base_quant_int2":
                    bits, sb, rb = 2, 0.20, 0.0
                    base = _make_model(args.model_id, care_cfg=None)
                    cc = _care_cfg(base, base_bits=bits, store_b=sb, read_b=rb)
                    os.environ["CAREKV_PREFILL_MODE"] = "base_quant"
                    model = patch_llama_model(base, cc); reset_all_caches(model)
                    prefill_mode = "base_quant"
                elif mode_label == "carekv_stored_int2":
                    bits, sb, rb = 2, 0.0, 0.0
                    base = _make_model(args.model_id, care_cfg=None)
                    cc = _care_cfg(base, base_bits=bits, store_b=sb, read_b=rb,
                                   optimized=True)
                    os.environ["CAREKV_PREFILL_MODE"]          = "carekv_stored"
                    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
                    os.environ["CAREKV_ROUTE_POLICY"]          = "joint"
                    os.environ["CAREKV_SCORE_NORMALIZE"]       = "1"
                    os.environ["CAREKV_CORRECTION_IMPL"]       = "cached"
                    os.environ["CAREKV_DEBUG_STATS"]           = "1"
                    model = patch_llama_model(base, cc); reset_all_caches(model)
                    prefill_mode = "carekv_stored_optimized"
                else:
                    print(f"[skip] unknown mode_label={mode_label}", flush=True)
                    continue

                reset_debug_stats()
                m = _measure(model, tok, prompt_len, effective_new)
                stats = get_debug_stats()
                row = dict(
                    model=args.model_id, mode=mode_label, base_bits=bits,
                    prefill_mode=prefill_mode, store_budget=sb, read_budget=rb,
                    prompt_len=prompt_len, new_tokens=effective_new,
                    prefill_ms=round(m["prefill_ms"], 2),
                    decode_ms_per_token=round(m["decode_ms_per_token"], 2),
                    tokens_per_sec=round(m["tokens_per_sec"], 3),
                    peak_gpu_mem_MB=round(m["peak_gpu_mem_MB"], 2),
                    K_reads=stats.get("k_slots_read", 0),
                    V_reads=stats.get("v_slots_read", 0),
                    correction_calls=stats.get("n_queries", 0),
                )
                rows.append(row)
                print(f"[bench] {mode_label:22s}  prompt={prompt_len:>5d}  "
                      f"prefill={m['prefill_ms']:>9.1f} ms  "
                      f"decode/tok={m['decode_ms_per_token']:>9.1f} ms  "
                      f"tok/s={m['tokens_per_sec']:>7.2f}  "
                      f"peak={m['peak_gpu_mem_MB']:>7.1f} MB",
                      flush=True)
            except Exception as e:
                print(f"[bench] {mode_label} prompt={prompt_len}: ERROR {type(e).__name__}: {e}",
                      flush=True)
                rows.append(dict(model=args.model_id, mode=mode_label, base_bits=-1,
                                 prefill_mode="?", store_budget=0, read_budget=0,
                                 prompt_len=prompt_len, new_tokens=0,
                                 prefill_ms=-1, decode_ms_per_token=-1,
                                 tokens_per_sec=-1, peak_gpu_mem_MB=-1))
            finally:
                try: del model
                except: pass
                gc.collect()
                if DEVICE == "cuda": torch.cuda.empty_cache()

    if rows:
        keys = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"\nwrote {len(rows)} rows → {args.out_csv}")
    else:
        print("no rows collected")


if __name__ == "__main__":
    main()
