"""baselines/kivi_style_quantizer.py — same-condition KIVI reimplementation.

KIVI's distinguishing trait:
  - K: **per-channel** quantization (one scale per kv-head per channel,
       taken across all tokens in the sequence)
  - V: **per-token**  quantization (one scale per kv-head per token,
       taken across all channels)

Faithful to KIVI's quantization scheme; does NOT use KIVI's CUDA kernels.
Same standalone logic as `baselines/kivi_style_adapter.py` (which
monkey-patches HF LlamaAttention); this version is an instance of
`BaseKVQuantizer` so it can plug into the CARE-KV residual pipeline.

Used when `CAREKV_BASE_QUANTIZER=kivi_style`. As of this turn, the
CARE-KV pipeline integration to use this quantizer for residual
computation is **deferred** (see "Integration status" in
summaries/carekv_on_base_quantizers.md); this class is exercised via the
existing `KIVIStyleAdapter` for stand-alone KIVI cells.
"""
from __future__ import annotations
from typing import Dict, Any, Optional

import torch
from torch import Tensor

from .quantizer_base import BaseKVQuantizer, BaseKVQuantResult
# Re-export the core helpers so existing callers of the underscore-
# prefixed names continue to work; single source of truth is
# care_kv/kivi_helpers.py.
from ..kivi_helpers import (
    quant_dequant_kivi_k as _quant_dequant_per_channel_K,
    quant_dequant_kivi_v as _quant_dequant_per_token_V,
)


class KIVIStyleQuantizer(BaseKVQuantizer):
    """Same-condition KIVI K/V quantization (no CUDA kernels)."""

    supports_post_rope_k = True
    supports_pre_rope_k = False
    supports_variable_bits = False

    def __init__(self, bits_k: int = 3, bits_v: int = 3):
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.name = f"kivi_style_INT{bits_k}K_INT{bits_v}V"

    def encode_kv(self, K_fp: Tensor, V_fp: Tensor,
                  metadata: Optional[Dict[str, Any]] = None) -> BaseKVQuantResult:
        K_hat = _quant_dequant_per_channel_K(K_fp, self.bits_k)
        V_hat = _quant_dequant_per_token_V(V_fp, self.bits_v)
        return BaseKVQuantResult(
            K_hat=K_hat, V_hat=V_hat,
            K_codes=None, V_codes=None,    # standalone returns dequantized only
            K_scale=None, V_scale=None,
            scheme_k=f"per-channel (scale across tokens), INT{self.bits_k} symmetric",
            scheme_v=f"per-token (scale across channels), INT{self.bits_v} symmetric",
            notes=("KIVI-style same-condition reimplementation. "
                   "Quantization scheme faithful to published KIVI K/V quant; "
                   "does NOT use KIVI's CUDA kernels. Label as "
                   "'KIVI-style same-condition reimplementation', NOT 'official KIVI'."),
        )

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64) -> Dict[str, float]:
        # K: bits_k per element + fp16 scale per (kv-head, channel)
        # V: bits_v per element + fp16 scale per (kv-head, token)
        k_bytes = (seq_len * hkv * head_dim * self.bits_k / 8.0
                    + 2 * hkv * head_dim)
        v_bytes = (seq_len * hkv * head_dim * self.bits_v / 8.0
                    + 2 * hkv * seq_len)
        total_mb = num_layers * (k_bytes + v_bytes) / (1024 * 1024)
        fp16_mb = num_layers * 2 * seq_len * hkv * head_dim * 2 / (1024 * 1024)
        return dict(
            estimated_kv_memory_MB=round(total_mb, 4),
            vs_fp16_kv_memory_ratio=round(total_mb / max(fp16_mb, 1e-9), 4),
            k_bytes_per_layer=round(k_bytes, 1),
            v_bytes_per_layer=round(v_bytes, 1),
        )
