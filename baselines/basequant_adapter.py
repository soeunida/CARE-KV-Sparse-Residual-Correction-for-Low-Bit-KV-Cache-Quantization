"""baselines/basequant_adapter.py — CARE-KV's base_quant path (no residual)."""
from __future__ import annotations
import os
import torch

from transformers import LlamaForCausalLM
from CARE_KV.care_kv import CacheConfig, patch_llama_model, reset_all_caches
from CARE_KV.care_kv.cache import apply_carekv_env_overrides
from .common import KVMethodAdapter, DEVICE, fp16_kv_mb


class BaseQuantAdapter(KVMethodAdapter):
    name = "base_quant_INT3"
    family = "base_quant"
    is_official = False
    is_reimplementation = True   # group_size symmetric base quant — a generic reference
    bit_width = "INT3"
    k_quant_scheme = "per-channel-group=32, symmetric"
    v_quant_scheme = "per-channel-group=32, symmetric"

    def __init__(self, bits: int = 3, group_size: int = 32):
        self.bits = bits
        self.group_size = group_size
        self.bit_width = f"INT{bits}"
        self.name = f"base_quant_INT{bits}"
        if group_size != 32:
            self.name += f"_gs{group_size}"
        self.k_quant_scheme = f"per-group={group_size}, symmetric"
        self.v_quant_scheme = f"per-group={group_size}, symmetric"

    def setup_model(self, model_id: str):
        env = dict(
            CAREKV_PREFILL_MODE="base_quant",
            CAREKV_BASE_BITS=str(self.bits),
            CAREKV_GROUP_SIZE=str(self.group_size),
            CAREKV_PACKED_BASE="1",
            CAREKV_SCALE_QUANT="int8",
            CAREKV_STORE_BUDGET_MODE="absolute",
            CAREKV_READ_BUDGET_MODE="absolute",
            CAREKV_STORE_ABS_K="0", CAREKV_STORE_ABS_V="0",
            CAREKV_READ_ABS_K="0",  CAREKV_READ_ABS_V="0",
            CAREKV_DEBUG_STATS="1",
        )
        for k, v in env.items():
            os.environ[k] = v
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map=DEVICE if DEVICE == "cuda" else None,
        )
        m.config.use_cache = False
        cfg = m.config
        hd = cfg.hidden_size // cfg.num_attention_heads
        kw = dict(
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd, base_bits=self.bits,
            group_size=self.group_size, k_channel_group=32, page_size=16, max_pages=512,
            v_token_block=4, sketch_dim=16,
            store_budget_ratio=0.0, read_budget_ratio=0.0,
            store_budget_mode="absolute", read_budget_mode="absolute",
        )
        apply_carekv_env_overrides(kw)
        cc = CacheConfig(**kw)
        m = patch_llama_model(m, cc)
        reset_all_caches(m)
        m.eval()
        return m

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64):
        fp16 = fp16_kv_mb(seq_len, num_layers, hkv, head_dim)
        ratio = self.bits / 16.0
        return dict(estimated_kv_memory_MB=fp16 * ratio,
                    estimated_total_cache_memory_MB=fp16 * ratio,
                    vs_fp16_kv_memory_ratio=ratio)

    def notes(self) -> str:
        return (f"CARE-KV base_quant prefill, INT{self.bits}, group_size={self.group_size}. "
                f"No residual correction.")
