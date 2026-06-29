"""baselines/quantizer_base.py — base-quantizer interface for CARE-KV.

Goal: let CARE-KV use ANY base KV quantizer (uniform per-group, KIVI-style
asymmetric, KVQuant-style pre-RoPE, etc.) as the source of K_hat / V_hat,
with CARE-KV's residual correction computed against that K_hat / V_hat.

General formula:
    R_K = K_fp - K_hat_external
    R_V = V_fp - V_hat_external
    O_care = O_base + ΔO_K + ΔO_V

A `BaseKVQuantizer` implementation is responsible for:
  - K_hat = quant_dequant(K_fp) under the method's K scheme
  - V_hat = quant_dequant(V_fp) under the method's V scheme
  - memory accounting that reflects what the method WOULD store on a real
    deployment (not what CARE-KV's cache actually stores in the experimental
    harness — see "Integration status" in
    summaries/carekv_on_base_quantizers.md).

This file ships the interface + result dataclass only. Concrete
quantizers live in `uniform_quantizer.py` and `kivi_style_quantizer.py`.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
from torch import Tensor


@dataclass
class BaseKVQuantResult:
    """One pair of K_hat / V_hat plus the metadata CARE-KV needs to
    compute residuals against and the memory accounting needs to report.

    Fields:
      K_hat / V_hat : the dequantized values that CARE-KV's residual is
                       computed against (`R = K_fp - K_hat`).
      K_codes / V_codes : encoded representation (int8 / packed); may be
                          None for adapters that only return K_hat / V_hat.
      K_scale / V_scale : per-method scale tensors; shape depends on the
                          quantization scheme (per-group / per-channel /
                          per-token / per-page-block).
      memory_bytes      : theoretical memory cost of the quantized
                          representation (codes + scales), used for the
                          "estimated KV memory MB" column.
      scheme_k / scheme_v : free-text labels ("per-channel INT3" etc.).
      notes             : caveats — e.g., "INT2 INSTABLE on TinyLlama".
    """
    K_hat: Tensor
    V_hat: Tensor
    K_codes: Optional[Tensor] = None
    V_codes: Optional[Tensor] = None
    K_scale: Optional[Tensor] = None
    V_scale: Optional[Tensor] = None
    memory_bytes: int = 0
    scheme_k: str = ""
    scheme_v: str = ""
    notes: str = ""


class BaseKVQuantizer:
    """Abstract base class. Subclass and implement `encode_kv`.

    Subclasses MUST be stateless / pure functions (given the same K_fp,
    V_fp, and metadata, return the same K_hat, V_hat). Statefulness like
    calibration scalars should live in the subclass's __init__.
    """

    name: str = "<unset>"
    bits_k: int = 16
    bits_v: int = 16
    supports_post_rope_k: bool = True
    supports_pre_rope_k: bool = False
    supports_variable_bits: bool = False  # per-token bit-width

    def encode_kv(self, K_fp: Tensor, V_fp: Tensor,
                  metadata: Optional[Dict[str, Any]] = None) -> BaseKVQuantResult:
        """Quantize K and V according to this method's scheme.

        Args:
          K_fp / V_fp : full-precision tensors, shape (B, Hkv, T, D) or (T, D).
          metadata    : optional per-call hints (e.g., layer id, RoPE state).
                        Implementations should default sensibly when None.
        Returns BaseKVQuantResult with K_hat / V_hat at least.
        """
        raise NotImplementedError

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64) -> Dict[str, float]:
        """Per-method theoretical memory (codes + scales) in MB.

        Default implementation: fp16 KV (subclasses override).
        """
        fp16_mb = num_layers * 2 * seq_len * hkv * head_dim * 2 / (1024 * 1024)
        return dict(
            estimated_kv_memory_MB=fp16_mb,
            vs_fp16_kv_memory_ratio=1.0,
            k_bytes_per_layer=2 * seq_len * hkv * head_dim,
            v_bytes_per_layer=2 * seq_len * hkv * head_dim,
        )

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(name={self.name!r}, "
                f"bits_k={self.bits_k}, bits_v={self.bits_v}, "
                f"pre_rope_k={self.supports_pre_rope_k}, "
                f"variable_bits={self.supports_variable_bits})")
