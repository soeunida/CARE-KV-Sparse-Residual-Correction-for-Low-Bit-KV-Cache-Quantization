"""
tools/paper_eval.py
-------------------
Single-file paper-evaluation harness for CARE-KV.

Subcommands:
    core-ppl       — FP/base_quant/carekv_eval/carekv_stored PPL across configs
    invariant      — R=0 / READ_BUDGET=0 ≡ base_quant exact-match check
    budget-sweep   — store_budget × read_budget grid (carekv_stored, V-only)
    vk-ablation    — v / k / both ablation at fixed budgets (INT3)
    memory         — actual + estimator memory across SEQ_LEN
    generation     — text-generation sanity (USE_CACHE=0 by default)
    figures        — diagnostic plots (compact set: layers 0, mid, last; INT3)

Each subcommand writes one CSV (or one image set) into the directory passed
via --out.  Designed to load the underlying HF model at most once per
(base_bits, packed_base) family.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, "/home/soeun")
from CARE_KV.care_kv import (
    CacheConfig, CAREKVCache, patch_llama_model, reset_all_caches,
    estimate_memory_bytes, get_debug_stats, reset_debug_stats,
    packed_row_bytes, apply_carekv_env_overrides,
)


MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────

def _silence():
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import warnings; warnings.filterwarnings("ignore")


def _build_tokenized(seq_len: int, repeat_text: int = 20):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None or tok.pad_token_id < 0:
        tok.pad_token_id = tok.eos_token_id or 0
    text = (
        "KV cache quantization reduces memory usage during autoregressive "
        "decoding of large language models. " * repeat_text
    )
    enc = tok(text, return_tensors="pt", truncation=True, max_length=seq_len)
    return tok, enc["input_ids"].to(DEVICE)


def _make_model(care_cfg: Optional[CacheConfig] = None):
    """Load TinyLlama and optionally apply the CARE-KV patch.  Returns model."""
    from transformers import LlamaForCausalLM
    torch.manual_seed(0)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    model.config.use_cache = False
    model.generation_config.use_cache = False
    if care_cfg is not None:
        model = patch_llama_model(model, care_cfg)
        reset_all_caches(model)
    model.eval()
    return model


def _care_cfg_for_model(model, *, base_bits: int, store_b: float, read_b: float,
                       packed_base: bool = True,
                       scale_dtype: str = "fp16", scale_quant: str = "none",
                       page_size: int = 16, group_size: int = 32,
                       k_channel_group: int = 32, v_token_block: int = 4,
                       sketch_dim: int = 16) -> CacheConfig:
    c = model.config
    hd = c.hidden_size // c.num_attention_heads
    return CacheConfig(
        num_layers=c.num_hidden_layers,
        num_heads=c.num_attention_heads,
        num_kv_heads=c.num_key_value_heads,
        head_dim=hd,
        base_bits=base_bits,
        page_size=page_size,
        max_pages=max(64, 8 * (1024 // page_size)),  # conservative
        group_size=group_size,
        k_channel_group=k_channel_group,
        v_token_block=v_token_block,
        store_budget_ratio=store_b,
        read_budget_ratio=read_b,
        sketch_dim=sketch_dim,
        packed_base=packed_base,
        scale_dtype=scale_dtype,
        scale_quant=scale_quant,
    )


def _ppl_from_loss(loss_value: float) -> float:
    return math.exp(loss_value)


def _set_env(prefill_mode: str, kind: str = "both", k_scale: float = 0.1,
             v_score: str = "output_aware", score_normalize: bool = False,
             debug_stats: bool = False):
    os.environ["CAREKV_PREFILL_MODE"] = prefill_mode
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = kind
    os.environ["CAREKV_K_CORRECTION_SCALE"] = str(k_scale)
    os.environ["CAREKV_V_SCORE"] = v_score
    os.environ["CAREKV_SCORE_NORMALIZE"] = "1" if score_normalize else "0"
    os.environ["CAREKV_DEBUG_STATS"] = "1" if debug_stats else "0"


# ─────────────────────────────────────────────
# core-ppl
# ─────────────────────────────────────────────

def cmd_core_ppl(args):
    _silence()
    tok, input_ids = _build_tokenized(args.seq_len)
    rows = []

    # (label, base_bits, prefill_mode, store_b, read_b, kind, k_scale, v_score, score_norm)
    configs = [
        ("fp",                    16, "fp",            0.0, 0.0,  "both", 0.1, "output_aware", False),
        ("base_quant_int4",        4, "base_quant",    0.0, 0.0,  "both", 0.1, "output_aware", False),
        ("base_quant_int3",        3, "base_quant",    0.0, 0.0,  "both", 0.1, "output_aware", False),
        ("base_quant_int2",        2, "base_quant",    0.0, 0.0,  "both", 0.1, "output_aware", False),
        ("carekv_eval_int3_v_R005",       3, "carekv_eval",   0.10, 0.05, "v",    0.1, "output_aware", False),
        ("carekv_stored_int3_v",          3, "carekv_stored", 0.10, 0.03, "v",    0.1, "output_aware", False),
        ("carekv_stored_int2_v",          2, "carekv_stored", 0.20, 0.05, "v",    0.1, "output_aware", False),
        ("carekv_stored_int3_both",       3, "carekv_stored", 0.10, 0.03, "both", 0.05,"output_aware", True),
    ]

    for label, bits, mode, sb, rb, kind, k_scale, vs, sn in configs:
        print(f"[core-ppl] {label} ...", flush=True)
        if mode == "fp" and bits == 16:
            # Plain fp16 LlamaForCausalLM, no patching
            from transformers import LlamaForCausalLM
            torch.manual_seed(0)
            model = LlamaForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16,
                device_map=DEVICE if DEVICE == "cuda" else None,
            )
            model.config.use_cache = False
            model.eval()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            dt = time.perf_counter() - t0
            ppl = _ppl_from_loss(out.loss.item())
            del model; torch.cuda.empty_cache()
            rows.append(dict(label=label, base_bits=16, prefill_mode="fp",
                             store_budget=0.0, read_budget=0.0, kind="-",
                             k_scale=0.0, v_score="-", score_normalize=False,
                             ppl=ppl, seconds=dt, current_clean_run=True,
                             v_slots_read=0, k_slots_read=0))
            continue

        _set_env(mode, kind=kind, k_scale=k_scale, v_score=vs,
                 score_normalize=sn, debug_stats=True)
        model = _make_model(care_cfg=None)   # build CARE-KV patched model
        care_cfg = _care_cfg_for_model(model, base_bits=bits, store_b=sb, read_b=rb,
                                       packed_base=True)
        model = patch_llama_model(model, care_cfg); reset_all_caches(model)
        reset_debug_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        dt = time.perf_counter() - t0
        ppl = _ppl_from_loss(out.loss.item())
        st = get_debug_stats()
        rows.append(dict(label=label, base_bits=bits, prefill_mode=mode,
                         store_budget=sb, read_budget=rb, kind=kind,
                         k_scale=k_scale, v_score=vs, score_normalize=sn,
                         ppl=ppl, seconds=dt, current_clean_run=True,
                         v_slots_read=st.get("v_slots_read", 0),
                         k_slots_read=st.get("k_slots_read", 0)))
        print(f"    PPL={ppl:.4f}  ({dt:.1f} s)  V={st.get('v_slots_read',0)} K={st.get('k_slots_read',0)}", flush=True)
        del model; torch.cuda.empty_cache()

    _write_csv(args.out_csv, rows)


# ─────────────────────────────────────────────
# invariant
# ─────────────────────────────────────────────

def cmd_invariant(args):
    _silence()
    _, input_ids = _build_tokenized(args.seq_len)
    rows = []
    for bits in [3, 2]:
        # base_quant baseline
        _set_env("base_quant")
        model = _make_model()
        cc = _care_cfg_for_model(model, base_bits=bits, store_b=0.10, read_b=0.0,
                                 packed_base=True)
        model = patch_llama_model(model, cc); reset_all_caches(model)
        with torch.no_grad():
            ppl_bq = _ppl_from_loss(model(input_ids=input_ids, labels=input_ids,
                                          use_cache=False).loss.item())
        del model; torch.cuda.empty_cache()

        # carekv_stored READ_BUDGET=0
        _set_env("carekv_stored", kind="both")
        model = _make_model()
        cc = _care_cfg_for_model(model, base_bits=bits, store_b=0.10, read_b=0.0,
                                 packed_base=True)
        model = patch_llama_model(model, cc); reset_all_caches(model)
        with torch.no_grad():
            ppl_st0 = _ppl_from_loss(model(input_ids=input_ids, labels=input_ids,
                                           use_cache=False).loss.item())
        del model; torch.cuda.empty_cache()

        diff = abs(ppl_bq - ppl_st0)
        status = "PASS" if diff < 1e-4 else "FAIL"
        print(f"[invariant] INT{bits}: base_quant={ppl_bq:.6f} stored(R=0)={ppl_st0:.6f} "
              f"|diff|={diff:.2e} {status}")
        rows.append(dict(base_bits=bits, base_quant_ppl=ppl_bq,
                         stored_r0_ppl=ppl_st0, abs_diff=diff,
                         status=status, current_clean_run=True))
    _write_csv(args.out_csv, rows)


# ─────────────────────────────────────────────
# budget-sweep (carekv_stored, V-only)
# ─────────────────────────────────────────────

def cmd_budget_sweep(args):
    _silence()
    _, input_ids = _build_tokenized(args.seq_len)
    rows = []

    grids = {
        3: ([0.05, 0.10, 0.20], [0.01, 0.03, 0.05]),
        2: ([0.10, 0.20, 0.30], [0.03, 0.05, 0.10]),
    }
    bits_list = [3] if args.bits_only_3 else [3, 2]

    for bits in bits_list:
        stores, reads = grids[bits]
        for sb in stores:
            for rb in reads:
                if rb > sb:
                    continue
                _set_env("carekv_stored", kind="v", v_score="output_aware",
                         debug_stats=True)
                model = _make_model()
                cc = _care_cfg_for_model(model, base_bits=bits, store_b=sb, read_b=rb,
                                         packed_base=True)
                model = patch_llama_model(model, cc); reset_all_caches(model)
                reset_debug_stats()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                dt = time.perf_counter() - t0
                ppl = _ppl_from_loss(out.loss.item())
                st = get_debug_stats()
                mem = estimate_memory_bytes(cc, args.seq_len)
                row = dict(base_bits=bits, store_budget=sb, read_budget=rb,
                           ppl=ppl, seconds=dt,
                           v_slots_read=st.get("v_slots_read", 0),
                           k_slots_read=st.get("k_slots_read", 0),
                           total_MB=mem["total_bytes"]/1e6,
                           vs_fp16=mem["compression_vs_fp16"],
                           current_clean_run=True)
                rows.append(row)
                print(f"[budget-sweep] INT{bits} S={sb} R={rb}  PPL={ppl:.4f}  "
                      f"V={st.get('v_slots_read',0)} K={st.get('k_slots_read',0)}  "
                      f"({dt:.1f}s)", flush=True)
                del model; torch.cuda.empty_cache()
    _write_csv(args.out_csv, rows)


# ─────────────────────────────────────────────
# vk-ablation
# ─────────────────────────────────────────────

def cmd_vk_ablation(args):
    _silence()
    _, input_ids = _build_tokenized(args.seq_len)
    rows = []
    SB, RB = 0.10, 0.03

    # v
    for k_scale in [0.0]:
        _set_env("carekv_stored", kind="v", v_score="output_aware",
                 score_normalize=True, k_scale=k_scale, debug_stats=True)
        _run_one(rows, bits=3, sb=SB, rb=RB, label=f"v_only",
                 kind="v", k_scale=k_scale, input_ids=input_ids,
                 score_normalize=True)

    for ks in [0.01, 0.02, 0.05]:
        _set_env("carekv_stored", kind="k", v_score="output_aware",
                 score_normalize=True, k_scale=ks, debug_stats=True)
        _run_one(rows, bits=3, sb=SB, rb=RB, label=f"k_only_kscale_{ks}",
                 kind="k", k_scale=ks, input_ids=input_ids,
                 score_normalize=True)

    for ks in [0.01, 0.02, 0.05]:
        _set_env("carekv_stored", kind="both", v_score="output_aware",
                 score_normalize=True, k_scale=ks, debug_stats=True)
        _run_one(rows, bits=3, sb=SB, rb=RB, label=f"both_kscale_{ks}",
                 kind="both", k_scale=ks, input_ids=input_ids,
                 score_normalize=True)
    _write_csv(args.out_csv, rows)


def _run_one(rows, *, bits, sb, rb, label, kind, k_scale, input_ids, score_normalize):
    model = _make_model()
    cc = _care_cfg_for_model(model, base_bits=bits, store_b=sb, read_b=rb,
                             packed_base=True)
    model = patch_llama_model(model, cc); reset_all_caches(model)
    reset_debug_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    dt = time.perf_counter() - t0
    ppl = _ppl_from_loss(out.loss.item())
    st = get_debug_stats()
    rows.append(dict(label=label, base_bits=bits, kind=kind, k_scale=k_scale,
                     store_budget=sb, read_budget=rb,
                     v_score="output_aware", score_normalize=score_normalize,
                     ppl=ppl, seconds=dt,
                     v_slots_read=st.get("v_slots_read", 0),
                     k_slots_read=st.get("k_slots_read", 0),
                     current_clean_run=True))
    print(f"[ablation] {label:30s}  PPL={ppl:.4f}  V={st.get('v_slots_read',0)} K={st.get('k_slots_read',0)}  ({dt:.1f}s)", flush=True)
    del model; torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# memory
# ─────────────────────────────────────────────

def cmd_memory(args):
    """Actual + estimator memory across SEQ_LEN.  CPU-only, instant."""
    # Use a TinyLlama-shaped cfg directly (no model load needed).
    seq_lens = [128, 512, 2048, 8192]
    rows = []
    for packed in [False, True]:
        for sb in [0.10]:
            for sd, sq in [("fp16", "none"), ("fp16", "int8")]:
                for T in seq_lens:
                    cfg = CacheConfig(
                        num_layers=22, num_heads=32, num_kv_heads=4, head_dim=64,
                        page_size=16, max_pages=max(64, math.ceil(T/16)),
                        base_bits=3, group_size=32, k_channel_group=32,
                        store_budget_ratio=sb, read_budget_ratio=0.03,
                        sketch_dim=16, packed_base=packed,
                        scale_dtype=sd, scale_quant=sq,
                    )
                    cache = CAREKVCache(cfg, torch.device("cpu"))
                    actual = sum(
                        getattr(cache, n).element_size() * getattr(cache, n).numel()
                        for n in ["base_K_codes","base_V_codes","base_K_scale","base_V_scale",
                                  "valid_tokens","k_residual_buf","v_residual_buf",
                                  "k_residual_scale","v_residual_scale"]
                        if getattr(cache, n, None) is not None
                    )
                    if cache.base_K_scale_master is not None:
                        for n in ["base_K_scale_master","base_V_scale_master"]:
                            actual += getattr(cache, n).element_size() * getattr(cache, n).numel()
                    est = estimate_memory_bytes(cfg, T)
                    rows.append(dict(
                        seq_len=T, packed_base=packed, scale_dtype=sd, scale_quant=sq,
                        actual_MB=actual/1e6, estimator_MB=est["total_bytes"]/1e6,
                        fp16_MB=est["fp16_kv_bytes"]/1e6,
                        actual_vs_fp16=actual/est["fp16_kv_bytes"],
                        estimator_vs_fp16=est["compression_vs_fp16"],
                        base_code_MB=(est["base_K_code_bytes"]+est["base_V_code_bytes"])/1e6,
                        scale_MB=(est["base_K_scale_bytes"]+est["base_V_scale_bytes"])/1e6,
                        residual_MB=(est["residual_K_bytes"]+est["residual_V_bytes"])/1e6,
                        meta_MB=(est["metadata_bytes"]+est["error_norm_bytes"])/1e6,
                        sketch_MB=est["sketch_bytes"]/1e6,
                        current_clean_run=True,
                    ))
                    del cache
    _write_csv(args.out_csv, rows)
    print(f"[memory] wrote {len(rows)} rows to {args.out_csv}")


# ─────────────────────────────────────────────
# generation
# ─────────────────────────────────────────────

def cmd_generation(args):
    _silence()
    from transformers import AutoTokenizer, LlamaForCausalLM
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
    PROMPT = "The capital of France is"

    def gen(label: str, mode: Optional[str], bits: Optional[int],
            store_b: float = 0.10, read_b: float = 0.03, use_cache: bool = False):
        torch.manual_seed(0)
        model = LlamaForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16,
            device_map=DEVICE if DEVICE == "cuda" else None,
        )
        model.config.use_cache = use_cache
        model.generation_config.use_cache = use_cache
        model.generation_config.pad_token_id = tok.pad_token_id
        if mode is not None:
            os.environ["CAREKV_PREFILL_MODE"] = mode
            cc = _care_cfg_for_model(model, base_bits=bits, store_b=store_b,
                                     read_b=read_b, packed_base=True)
            model = patch_llama_model(model, cc); reset_all_caches(model)
        model.eval()
        inp = tok(PROMPT, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=24, do_sample=False,
                                 use_cache=use_cache, pad_token_id=tok.pad_token_id)
        txt = tok.decode(out[0], skip_special_tokens=True)
        path = os.path.join(args.out_dir, f"{label}.txt")
        with open(path, "w") as f:
            f.write(f"MODEL={MODEL_ID}\nPROMPT={PROMPT!r}\nMODE={mode}\nBITS={bits}\n")
            f.write(f"USE_CACHE={use_cache}\n----\n{txt}\n")
        print(f"[generation] {label}: {txt!r}", flush=True)
        del model; torch.cuda.empty_cache()

    gen("fp16",                   None,            None)
    gen("base_quant_int3",        "base_quant",    3)
    gen("carekv_stored_int3",     "carekv_stored", 3, store_b=0.10, read_b=0.03)

    if args.try_use_cache:
        try:
            gen("carekv_stored_int3_use_cache_true", "carekv_stored", 3,
                store_b=0.10, read_b=0.03, use_cache=True)
        except Exception as e:
            with open(os.path.join(args.out_dir, "use_cache_true_blocker.txt"), "w") as f:
                f.write(f"use_cache=True failed (Phase 4 not implemented):\n{repr(e)}\n")


# ─────────────────────────────────────────────
# figures (compact: 3 layers, INT3 only)
# ─────────────────────────────────────────────

def cmd_figures(args):
    _silence()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] matplotlib not available; skipping")
        return

    from transformers import AutoTokenizer, LlamaForCausalLM
    from CARE_KV.care_kv.quantizer import QuantConfig, quantize_and_residual, dequantize
    from CARE_KV.care_kv.cache import CAREKVCache
    from CARE_KV.care_kv.layer import CAREKVLayer
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    model.eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    text = "KV cache quantization. " * 30
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
    T = enc["input_ids"].shape[1]
    hidden_in = model.model.embed_tokens(enc["input_ids"])
    position_ids = torch.arange(T, device=hidden_in.device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(hidden_in, position_ids)

    layer_ids = [0, cfg.num_hidden_layers // 2, cfg.num_hidden_layers - 1]
    qcfg = QuantConfig(bits=3, group_size=32)

    for li in layer_ids:
        layer = model.model.layers[li]
        attn = layer.self_attn
        hn = layer.input_layernorm(hidden_in)

        Q = hn @ attn.q_proj.weight.T
        K_pre = hn @ attn.k_proj.weight.T
        V = hn @ attn.v_proj.weight.T
        Hq, Hkv, D = cfg.num_attention_heads, cfg.num_key_value_heads, head_dim
        Q = Q.reshape(1, T, Hq, D).permute(0, 2, 1, 3)
        K_pre_r = K_pre.reshape(1, T, Hkv, D).permute(0, 2, 1, 3)
        V_r = V.reshape(1, T, Hkv, D).permute(0, 2, 1, 3)

        _, K_post = apply_rotary_pos_emb(Q, K_pre_r, cos, sin)

        # Quantize K_post, V, compute R_K, R_V (post-RoPE)
        K_post0 = K_post[0, 0].float()    # (T, D) for KV head 0
        V0 = V_r[0, 0].float()
        K_pre0 = K_pre_r[0, 0].float()
        Kc, Ks, _, _ = quantize_and_residual(K_post0, qcfg)
        Vc, Vs, _, _ = quantize_and_residual(V0, qcfg)
        K_hat = dequantize(Kc, Ks, K_post0.shape, qcfg)
        V_hat = dequantize(Vc, Vs, V0.shape, qcfg)
        R_K = (K_post0 - K_hat).float().abs()
        R_V = (V0 - V_hat).float().abs()

        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes[0, 0].hist(K_pre0.float().detach().cpu().numpy().flatten(), bins=64, alpha=0.7, color="steelblue")
        axes[0, 0].set_title(f"K pre-RoPE (L{li})")
        axes[0, 1].hist(K_post0.float().detach().cpu().numpy().flatten(), bins=64, alpha=0.7, color="indianred")
        axes[0, 1].set_title(f"K post-RoPE (L{li})")
        axes[0, 2].hist(V0.float().detach().cpu().numpy().flatten(), bins=64, alpha=0.7, color="seagreen")
        axes[0, 2].set_title(f"V (L{li})")

        im1 = axes[1, 0].imshow(R_K.detach().cpu().numpy(), aspect="auto", cmap="hot")
        axes[1, 0].set_title(f"|R_K| heatmap (L{li}, INT3)")
        axes[1, 0].set_xlabel("channel"); axes[1, 0].set_ylabel("token")
        fig.colorbar(im1, ax=axes[1, 0])

        im2 = axes[1, 1].imshow(R_V.detach().cpu().numpy(), aspect="auto", cmap="hot")
        axes[1, 1].set_title(f"|R_V| heatmap (L{li}, INT3)")
        axes[1, 1].set_xlabel("channel"); axes[1, 1].set_ylabel("token")
        fig.colorbar(im2, ax=axes[1, 1])

        # Per-token V residual norm = V routing "norm" score
        axes[1, 2].plot(R_V.norm(dim=-1).detach().cpu().numpy(), label="||R_V_t||", color="seagreen")
        axes[1, 2].plot(R_K.norm(dim=-1).detach().cpu().numpy(), label="||R_K_t||", color="indianred")
        axes[1, 2].set_title(f"Per-token residual norms (L{li})")
        axes[1, 2].set_xlabel("token"); axes[1, 2].legend()

        plt.tight_layout()
        path = os.path.join(args.out_dir, f"layer_{li:02d}_diagnostics.png")
        plt.savefig(path, dpi=110)
        plt.close(fig)
        print(f"[figures] saved {path}", flush=True)

    del model; torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# memory-pareto (Phase D)
# ─────────────────────────────────────────────

def cmd_memory_pareto(args):
    """Sweep residual granularity (page_size, v_token_block, k_channel_group,
    sketch_dim) at fixed INT3 carekv_stored, V-only, store/read budgets that
    actually trigger reads.  Emits PPL + full memory breakdown + slot reads
    per config."""
    _silence()
    _, input_ids = _build_tokenized(args.seq_len)

    # (label, page_size, v_token_block, k_channel_group, sketch_dim, store_b, read_b)
    base_S, base_R = 0.20, 0.10
    configs = [
        ("baseline",      16, 4,  32, 16, base_S, base_R),
        ("vblock_8",      16, 8,  32, 16, base_S, base_R),
        ("vblock_16",     16, 16, 32, 16, base_S, base_R),
        ("kgroup_16",     16, 4,  16, 16, base_S, base_R),
        ("kgroup_64",     16, 4,  64, 16, base_S, base_R),
        ("sketch_8",      16, 4,  32,  8, base_S, base_R),
        ("sketch_32",     16, 4,  32, 32, base_S, base_R),
        ("page_8",         8, 4,  32, 16, base_S, base_R),
        ("page_32",       32, 4,  32, 16, base_S, base_R),
    ]

    rows = []
    for (label, ps, vtb, kcg, sd, sb, rb) in configs:
        _set_env("carekv_stored", kind="v", v_score="output_aware",
                 score_normalize=False, debug_stats=True)
        model = _make_model()
        try:
            cc = _care_cfg_for_model(
                model, base_bits=3, store_b=sb, read_b=rb,
                packed_base=True, page_size=ps, group_size=32,
                k_channel_group=kcg, v_token_block=vtb, sketch_dim=sd,
            )
            model = patch_llama_model(model, cc); reset_all_caches(model)
            reset_debug_stats()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            dt = time.perf_counter() - t0
            ppl = _ppl_from_loss(out.loss.item())
            st = get_debug_stats()
            mem = estimate_memory_bytes(cc, args.seq_len)

            nq = max(st.get("n_queries", 1), 1)
            actual_read_ratio = (st.get("v_slots_read", 0) + st.get("k_slots_read", 0)) / nq

            row = dict(
                label=label, page_size=ps, v_token_block=vtb,
                k_channel_group=kcg, sketch_dim=sd,
                store_budget=sb, read_budget=rb,
                base_bits=3,
                ppl=ppl, seconds=dt,
                v_slots_read=st.get("v_slots_read", 0),
                k_slots_read=st.get("k_slots_read", 0),
                n_queries=nq,
                actual_read_ratio=actual_read_ratio,
                total_MB=mem["total_bytes"]/1e6,
                vs_fp16=mem["compression_vs_fp16"],
                base_MB=(mem["base_K_code_bytes"]+mem["base_V_code_bytes"])/1e6,
                scale_MB=(mem["base_K_scale_bytes"]+mem["base_V_scale_bytes"])/1e6,
                residual_MB=(mem["residual_K_bytes"]+mem["residual_V_bytes"])/1e6,
                metadata_MB=(mem["metadata_bytes"]+mem["error_norm_bytes"])/1e6,
                sketch_MB=mem["sketch_bytes"]/1e6,
                current_clean_run=True,
            )
            rows.append(row)
            print(f"[mem-pareto] {label:14s} ps={ps:2d} vtb={vtb:2d} kcg={kcg:2d} sk={sd:2d} "
                  f"PPL={ppl:.4f} total_MB={mem['total_bytes']/1e6:.2f} "
                  f"vs_fp16={mem['compression_vs_fp16']:.3f}x "
                  f"V={st.get('v_slots_read',0)} K={st.get('k_slots_read',0)} "
                  f"({dt:.1f}s)", flush=True)
        except Exception as e:
            print(f"[mem-pareto] {label}: ERROR {repr(e)}", flush=True)
            rows.append(dict(label=label, page_size=ps, v_token_block=vtb,
                             k_channel_group=kcg, sketch_dim=sd, error=str(e),
                             current_clean_run=True))
        finally:
            del model
            torch.cuda.empty_cache()
    _write_csv(args.out_csv, rows)


# ─────────────────────────────────────────────
# CSV helper
# ─────────────────────────────────────────────

def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows → {path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("core-ppl")
    sp.add_argument("--seq-len", type=int, default=128)
    sp.add_argument("--out-csv", required=True)
    sp.set_defaults(func=cmd_core_ppl)

    sp = sub.add_parser("invariant")
    sp.add_argument("--seq-len", type=int, default=128)
    sp.add_argument("--out-csv", required=True)
    sp.set_defaults(func=cmd_invariant)

    sp = sub.add_parser("budget-sweep")
    sp.add_argument("--seq-len", type=int, default=128)
    sp.add_argument("--bits-only-3", action="store_true")
    sp.add_argument("--out-csv", required=True)
    sp.set_defaults(func=cmd_budget_sweep)

    sp = sub.add_parser("vk-ablation")
    sp.add_argument("--seq-len", type=int, default=128)
    sp.add_argument("--out-csv", required=True)
    sp.set_defaults(func=cmd_vk_ablation)

    sp = sub.add_parser("memory")
    sp.add_argument("--out-csv", required=True)
    sp.set_defaults(func=cmd_memory)

    sp = sub.add_parser("memory-pareto")
    sp.add_argument("--seq-len", type=int, default=128)
    sp.add_argument("--out-csv", required=True)
    sp.set_defaults(func=cmd_memory_pareto)

    sp = sub.add_parser("generation")
    sp.add_argument("--out-dir", required=True)
    sp.add_argument("--try-use-cache", action="store_true")
    sp.set_defaults(func=cmd_generation)

    sp = sub.add_parser("figures")
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=cmd_figures)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
