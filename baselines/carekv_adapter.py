"""baselines/carekv_adapter.py — CARE-KV paper-best (fixed or adaptive)."""
from __future__ import annotations
import os
import torch

from transformers import LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats,
)
from CARE_KV.care_kv.cache import apply_carekv_env_overrides
from .common import KVMethodAdapter, DEVICE, fp16_kv_mb, resolve_device_map


class CAREKVAdapter(KVMethodAdapter):
    family = "care_kv"
    is_official = False
    is_reimplementation = True   # paper-best is this codebase's own method
    bit_width = "INT3"
    uses_residual = True
    uses_query_aware_routing = True

    def __init__(self, mode: str = "fixed", rel_threshold: float = 0.0,
                 sk: int = 2, sv: int = 4, rk: int = 2, rv: int = 2,
                 max_rk: int = 4, max_rv: int = 4, bits: int = 3,
                 base_quantizer: str = "uniform",
                 bits_k: int = -1, bits_v: int = -1,
                 max_pages: int = 0, k_store_mode: str = "post_rope",
                 correction_impl: str = "cached",
                 k_channel_group: int = 32, v_token_block: int = 4):
        """mode: 'fixed' (RK=RV=2) or 'adaptive' (adaptive_score with rel_threshold).
        base_quantizer: 'uniform' (paper-best), 'kivi_style' (Phase Q-stacked),
                          'rotatekv_style' or 'kvquant_style'
                          (base-quantizer-expansion).
        k_store_mode: only meaningful for base_quantizer == 'kvquant_style'.
                          'post_rope' (default, KIVI-equivalent alias) or
                          'pre_rope' (true KVQuant: quantize K before RoPE,
                          re-apply RoPE to K_hat so the residual stays post-RoPE).
        bits_k / bits_v: per-kind override; -1 → fall back to `bits`."""
        self.mode = mode
        self.rel_threshold = rel_threshold
        self.sk, self.sv = sk, sv
        self.rk, self.rv = rk, rv
        self.max_rk, self.max_rv = max_rk, max_rv
        self.bits = bits
        self.base_quantizer = base_quantizer
        self.k_store_mode = k_store_mode
        self.correction_impl = correction_impl
        # Residual granularity (coverage lever): SK_cap = head_dim/k_channel_group,
        # SV_cap = ceil(page_size/v_token_block).
        self.k_channel_group = k_channel_group
        self.v_token_block = v_token_block
        self.bits_k = bits_k if bits_k > 0 else bits
        self.bits_v = bits_v if bits_v > 0 else bits
        # max_pages=0 → use default (512). The fp16 side buffer scales
        # as max_pages, so callers running on smaller sequences should
        # pass a tighter value to avoid OOM.
        self.max_pages = max_pages if max_pages > 0 else 512

        _SIDE = {"kivi_style", "rotatekv_style", "randrot_style", "kvquant_style"}
        _kvq_short = "KVQuantPreRoPE" if k_store_mode == "pre_rope" else "KVQuantPostRoPE"
        _rope_tag = "preRoPE" if k_store_mode == "pre_rope" else "postRoPE"
        _SHORT = {"kivi_style": "KIVI",
                   "rotatekv_style": f"RotateKV_{_rope_tag}",
                   "randrot_style": f"RandRot_{_rope_tag}",
                   "kvquant_style": _kvq_short}
        _FAMILY = {"kivi_style": "kivi_plus_carekv",
                    "rotatekv_style": "rotatekv_plus_carekv",
                    "randrot_style": "randrot_plus_carekv",
                    "kvquant_style": "kvquant_plus_carekv"}
        _LONG_K = {"kivi_style": "KIVI, scale across tokens",
                    "rotatekv_style": f"Hadamard-rotated ({_rope_tag}), scale across tokens",
                    "randrot_style": f"random-orthonormal-rotated ({_rope_tag}), scale across tokens",
                    "kvquant_style": "KVQuant post-RoPE variant, scale across tokens"}
        _LONG_V = {"kivi_style": "KIVI, scale across channels",
                    "rotatekv_style": f"Hadamard-rotated ({_rope_tag}), scale across channels",
                    "randrot_style": f"random-orthonormal-rotated ({_rope_tag}), scale across channels",
                    "kvquant_style": "KVQuant post-RoPE variant, scale across channels"}

        if base_quantizer in _SIDE:
            short = _SHORT[base_quantizer]
            tag = f"{short}_INT{self.bits_k}K_INT{self.bits_v}V_plus_CAREKV"
        else:
            tag = f"CAREKV_fixed_SK{sk}SV{sv}_RK{rk}RV{rv}"
        if mode == "adaptive":
            self.name = f"CAREKV_adaptive_rel{rel_threshold:.2f}"
            if base_quantizer in _SIDE:
                short = _SHORT[base_quantizer]
                self.name = f"{short}_INT{self.bits_k}K_INT{self.bits_v}V_plus_CAREKV_adapt{rel_threshold:.2f}"
        else:
            self.name = tag
        self.bit_width = (f"K=INT{self.bits_k} V=INT{self.bits_v} + CARE-KV residual"
                          if base_quantizer in _SIDE else f"INT{bits}")
        if base_quantizer in _SIDE:
            self.k_quant_scheme = (f"per-channel ({_LONG_K[base_quantizer]}), "
                                    f"INT{self.bits_k} + CARE-KV residual slots")
            self.v_quant_scheme = (f"per-token  ({_LONG_V[base_quantizer]}), "
                                    f"INT{self.bits_v} + CARE-KV residual slots")
            self.family = _FAMILY[base_quantizer]
        else:
            self.k_quant_scheme = "per-channel-group=32 + selected residual slots"
            self.v_quant_scheme = "per-channel-group=32 + selected residual slots"

    def setup_model(self, model_id: str):
        reset_debug_stats()
        env = dict(
            CAREKV_PREFILL_MODE="carekv_stored",
            CAREKV_PREFILL_RESIDUAL_KIND="both",
            CAREKV_ROUTE_POLICY="joint",
            CAREKV_SCORE_NORMALIZE="1",
            CAREKV_CORRECTION_IMPL=self.correction_impl,
            CAREKV_BUDGET_POLICY="uniform",
            CAREKV_PACKED_BASE="1",
            CAREKV_SCALE_QUANT="int8",
            CAREKV_BASE_BITS=str(self.bits),
            CAREKV_GROUP_SIZE="32",
            CAREKV_STORE_BUDGET_MODE="absolute",
            CAREKV_STORE_ABS_K=str(self.sk), CAREKV_STORE_ABS_V=str(self.sv),
            CAREKV_DEBUG_STATS="1",
            CAREKV_BASE_QUANTIZER=self.base_quantizer,
            CAREKV_K_BITS=str(self.bits_k),
            CAREKV_V_BITS=str(self.bits_v),
            CAREKV_K_STORE_MODE=self.k_store_mode,
            CAREKV_K_CHANNEL_GROUP=str(self.k_channel_group),
            CAREKV_V_TOKEN_BLOCK=str(self.v_token_block),
        )
        if self.mode == "adaptive":
            env.update(
                CAREKV_READ_BUDGET_MODE="adaptive_score",
                CAREKV_READ_ABS_K=str(self.max_rk),
                CAREKV_READ_ABS_V=str(self.max_rv),
                CAREKV_READ_RELATIVE_THRESHOLD=str(self.rel_threshold),
                CAREKV_READ_ABSOLUTE_THRESHOLD="0.0",
                CAREKV_READ_MIN_KEEP="0",
            )
        else:
            env.update(
                CAREKV_READ_BUDGET_MODE="absolute",
                CAREKV_READ_ABS_K=str(self.rk), CAREKV_READ_ABS_V=str(self.rv),
                CAREKV_READ_RELATIVE_THRESHOLD="0.0",
                CAREKV_READ_ABSOLUTE_THRESHOLD="0.0",
                CAREKV_READ_MIN_KEEP="0",
            )
        for k, v in env.items():
            os.environ[k] = v
        torch.manual_seed(0)
        # device_map via CAREKV_DEVICE_MAP=auto shards >20B models across the
        # GPUs exposed by CUDA_VISIBLE_DEVICES (see resolve_device_map).
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map=resolve_device_map(), low_cpu_mem_usage=True,
        )
        m.config.use_cache = False
        cfg = m.config
        hd = cfg.hidden_size // cfg.num_attention_heads
        kw = dict(
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd, base_bits=self.bits,
            group_size=32, k_channel_group=32, page_size=16,
            max_pages=self.max_pages,
            v_token_block=4, sketch_dim=32,   # full-rank sketch (=k_channel_group)
            store_budget_ratio=0.0, read_budget_ratio=0.0,
            store_budget_mode="absolute", read_budget_mode="absolute",
        )
        apply_carekv_env_overrides(kw)
        cc = CacheConfig(**kw)
        m = patch_llama_model(m, cc)
        reset_all_caches(m)
        m.eval()
        return m

    def collect_debug_stats(self):
        s = get_debug_stats()
        return dict(
            k_reads=int(s.get("k_slots_read", 0)),
            v_reads=int(s.get("v_slots_read", 0)),
            stored_k_slots=int(s.get("k_slots_stored", 0)),
            stored_v_slots=int(s.get("v_slots_stored", 0)),
        )

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64):
        """Memory model:

        - **uniform** base: per-group INT3 packed + int8 page scales
          ≈ self.bits/16 of fp16 KV. CARE-KV residual slots add a small
          fixed overhead (~5%).
        - **kivi_style** base: per-channel K (bits_k) + per-token V
          (bits_v) + fp16 scale headers. CARE-KV residual slots add
          ~5% over the base. The fp16 side-buffer in the cache
          (prototype-only) is NOT counted — only KIVI's theoretical
          bits, as that's what a production implementation would store.
        """
        fp16 = fp16_kv_mb(seq_len, num_layers, hkv, head_dim)
        residual_overhead = 0.05            # SK=2/16 K + SV=4/16 V at 4-bit packed
        if self.base_quantizer in {"kivi_style", "rotatekv_style", "kvquant_style"}:
            # Per-channel K + per-token V theoretical per-layer bytes
            # (matches KIVIStyleQuantizer.estimate_memory). The
            # Hadamard rotation in rotatekv_style adds no per-token
            # cost — H is a fixed constant shared across the whole
            # cache. The kvquant_style post-RoPE variant uses the same
            # layout as KIVI.
            k_bytes = (seq_len * hkv * head_dim * self.bits_k / 8.0
                        + 2 * hkv * head_dim)            # fp16 K scale per channel
            v_bytes = (seq_len * hkv * head_dim * self.bits_v / 8.0
                        + 2 * hkv * seq_len)              # fp16 V scale per token
            base_mb = num_layers * (k_bytes + v_bytes) / (1024 * 1024)
        else:
            base_mb = fp16 * (self.bits / 16.0)
        residual_mb = fp16 * residual_overhead
        total_mb = base_mb + residual_mb
        return dict(
            estimated_kv_memory_MB=round(total_mb, 4),
            estimated_total_cache_memory_MB=round(total_mb, 4),
            vs_fp16_kv_memory_ratio=round(total_mb / max(fp16, 1e-9), 4),
            base_memory_MB=round(base_mb, 4),
            residual_memory_MB=round(residual_mb, 4),
            base_quantizer=self.base_quantizer,
        )

    def effective_budgets(self):
        if self.mode == "adaptive":
            return dict(
                effective_store_budget=f"SK={self.sk} SV={self.sv}",
                effective_read_budget=f"adaptive max RK={self.max_rk} RV={self.max_rv} rel={self.rel_threshold:.2f}",
            )
        return dict(
            effective_store_budget=f"SK={self.sk} SV={self.sv}",
            effective_read_budget=f"fixed RK={self.rk} RV={self.rv}",
        )

    def notes(self) -> str:
        _kvq_tag = (
            "KVQuant-style base quant (TRUE pre-RoPE: K quantized before "
            "RoPE on the smoother unrotated per-channel distribution, then "
            "K_hat re-rotated so the CARE-KV residual is computed post-RoPE; "
            "per-channel K + per-token V)"
            if self.k_store_mode == "pre_rope" else
            "KVQuant-style base quant (POST-RoPE variant — quantizes the "
            "rotated K, equivalent to KIVI; per-channel K + per-token V)"
        )
        _BASE_TAGS = {
            "kivi_style": "KIVI-style base quant (per-channel K + per-token V)",
            "rotatekv_style": "RotateKV-style base quant (Walsh-Hadamard rotation + per-channel K + per-token V + inverse rotation)",
            "kvquant_style": _kvq_tag,
        }
        if self.base_quantizer in _BASE_TAGS:
            base = (f"{_BASE_TAGS[self.base_quantizer]}, INT{self.bits_k}K / "
                    f"INT{self.bits_v}V, same-condition reimpl + CARE-KV residual "
                    f"correction (carekv_stored, joint+normalize, cached, "
                    f"SK={self.sk} SV={self.sv} RK={self.rk} RV={self.rv}). "
                    "K_hat/V_hat held in fp16 side-buffer in this prototype; "
                    "memory accounting reports the base scheme's theoretical bits.")
            if self.mode == "adaptive":
                return base + f" Read-budget mode: adaptive_score rel={self.rel_threshold}."
            return base
        if self.mode == "adaptive":
            return (f"CARE-KV paper-best + adaptive_score read budget "
                    f"(rel_threshold={self.rel_threshold}). Confirmed on WT-2 N=4 at rel=0.05.")
        return ("CARE-KV paper-best (carekv_stored, joint+normalize, cached, "
                f"SK={self.sk} SV={self.sv} RK={self.rk} RV={self.rv}).")
