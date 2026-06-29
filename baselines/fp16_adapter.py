"""baselines/fp16_adapter.py — reference (no KV compression)."""
from __future__ import annotations
import torch

from transformers import LlamaForCausalLM
from .common import KVMethodAdapter, DEVICE, fp16_kv_mb


class FP16Adapter(KVMethodAdapter):
    name = "fp16"
    family = "fp16"
    is_official = False
    is_reimplementation = False  # not a reimpl — it's the unmodified reference
    bit_width = "fp16"
    k_quant_scheme = "fp16 (none)"
    v_quant_scheme = "fp16 (none)"

    def setup_model(self, model_id: str):
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map=DEVICE if DEVICE == "cuda" else None,
        )
        m.config.use_cache = False
        m.eval()
        return m

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64):
        fp16 = fp16_kv_mb(seq_len, num_layers, hkv, head_dim)
        return dict(estimated_kv_memory_MB=fp16,
                    estimated_total_cache_memory_MB=fp16,
                    vs_fp16_kv_memory_ratio=1.0)

    def notes(self) -> str:
        return "Unmodified HF LlamaForCausalLM at fp16 (reference)."
