"""baselines/uniform_quantizer.py — wraps CARE-KV's existing per-group quant.

This is the "current behavior" reference: per-group symmetric
quantization along the last dim, group_size=32 by default. Both K and V
use the same scheme.

Used when `CAREKV_BASE_QUANTIZER=uniform` (the default).
"""
from __future__ import annotations
from typing import Dict, Any, Optional

import torch
from torch import Tensor

from CARE_KV.care_kv.quantizer import QuantConfig, quantize_and_residual, dequantize
from .quantizer_base import BaseKVQuantizer, BaseKVQuantResult


class UniformBaseQuantizer(BaseKVQuantizer):
    """Per-group symmetric quantization along the last dim (CARE-KV default)."""

    supports_post_rope_k = True
    supports_pre_rope_k = False
    supports_variable_bits = False

    def __init__(self, bits_k: int = 3, bits_v: int = 3, group_size: int = 32):
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.group_size = group_size
        self.name = f"uniform_INT{bits_k}K_INT{bits_v}V_gs{group_size}"
        if bits_k == bits_v:
            self.name = f"uniform_INT{bits_k}_gs{group_size}"
        self._qcfg_k = QuantConfig(bits=bits_k, group_size=group_size)
        self._qcfg_v = QuantConfig(bits=bits_v, group_size=group_size)

    def _qd(self, x: Tensor, cfg: QuantConfig):
        codes, scale, _, _ = quantize_and_residual(x, cfg)
        x_hat = dequantize(codes, scale, x.shape, cfg).to(x.dtype)
        return codes, scale, x_hat

    def encode_kv(self, K_fp: Tensor, V_fp: Tensor,
                  metadata: Optional[Dict[str, Any]] = None) -> BaseKVQuantResult:
        K_codes, K_scale, K_hat = self._qd(K_fp, self._qcfg_k)
        V_codes, V_scale, V_hat = self._qd(V_fp, self._qcfg_v)
        return BaseKVQuantResult(
            K_hat=K_hat, V_hat=V_hat,
            K_codes=K_codes, V_codes=V_codes,
            K_scale=K_scale, V_scale=V_scale,
            scheme_k=f"per-group={self.group_size}, INT{self.bits_k}, symmetric",
            scheme_v=f"per-group={self.group_size}, INT{self.bits_v}, symmetric",
            notes="UniformBaseQuantizer — wraps CARE-KV's existing per-group quant path.",
        )

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64) -> Dict[str, float]:
        # Per-group: ceil(D / group_size) scales per token per kv-head.
        groups_per_token = max(1, head_dim // self.group_size)
        # K bytes / layer: codes (T * Hkv * D bits) + scales (T * Hkv * groups * 2 bytes)
        k_bytes = (seq_len * hkv * head_dim * self.bits_k / 8.0
                    + seq_len * hkv * groups_per_token * 2)
        v_bytes = (seq_len * hkv * head_dim * self.bits_v / 8.0
                    + seq_len * hkv * groups_per_token * 2)
        total_mb = num_layers * (k_bytes + v_bytes) / (1024 * 1024)
        fp16_mb = num_layers * 2 * seq_len * hkv * head_dim * 2 / (1024 * 1024)
        return dict(
            estimated_kv_memory_MB=round(total_mb, 4),
            vs_fp16_kv_memory_ratio=round(total_mb / max(fp16_mb, 1e-9), 4),
            k_bytes_per_layer=round(k_bytes, 1),
            v_bytes_per_layer=round(v_bytes, 1),
        )
