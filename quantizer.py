"""
care_kv/quantizer.py
--------------------
INT2 / INT3 / INT4 symmetric per-group scalar quantizer for KV cache.

Supports:
  - forward:  fp16/bf16 tensor  →  (int_codes, scale)
  - backward: (int_codes, scale) →  fp16/bf16 tensor  (dequantize)
  - residual: original - dequantized  →  R_K / R_V
"""

from __future__ import annotations
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QuantConfig:
    bits: int = 2            # 2, 3, or 4
    group_size: int = 64     # channels per group (for K per-channel, V per-token)
    symmetric: bool = True   # symmetric around 0
    clip_ratio: float = 1.0  # scale = clip_ratio * max_abs (1.0 = no clip)


# ─────────────────────────────────────────────
# Core quantize / dequantize
# ─────────────────────────────────────────────

def quantize(x: Tensor, cfg: QuantConfig) -> Tuple[Tensor, Tensor]:
    """
    Quantize x to `cfg.bits` bits per element.

    x shape: (..., D)   last dim is quantized in groups of cfg.group_size

    Returns
    -------
    codes : IntTensor  same leading dims, last dim = D   (values in [-2^(b-1), 2^(b-1)-1])
    scale : FloatTensor leading dims, last dim = D // group_size
    """
    D = x.shape[-1]
    g = cfg.group_size
    assert D % g == 0, f"head_dim {D} must be divisible by group_size {g}"

    orig_shape = x.shape
    # Reshape to (..., num_groups, group_size)
    xg = x.reshape(*orig_shape[:-1], D // g, g)

    # Compute per-group scale
    max_abs = xg.abs().amax(dim=-1, keepdim=True)  # (..., G, 1)
    max_abs = (max_abs * cfg.clip_ratio).clamp(min=1e-8)

    qmax = 2 ** (cfg.bits - 1) - 1   # e.g. 1 for 2-bit, 3 for 3-bit, 7 for 4-bit
    scale = max_abs / qmax            # (..., G, 1)

    # Quantize
    xq = (xg / scale).round().clamp(-qmax - 1, qmax)
    codes = xq.to(torch.int8).reshape(orig_shape)

    # scale shape: (..., G)
    scale_out = scale.squeeze(-1)  # (..., G)

    return codes, scale_out


def dequantize(codes: Tensor, scale: Tensor, orig_shape: Tuple, cfg: QuantConfig) -> Tensor:
    """
    Reconstruct fp tensor from (codes, scale).

    codes: IntTensor (..., D)
    scale: FloatTensor (..., D // group_size)
    orig_shape: shape of the original fp tensor
    """
    D = orig_shape[-1]
    g = cfg.group_size

    # (..., G, g)
    xg = codes.reshape(*orig_shape[:-1], D // g, g).float()
    s  = scale.unsqueeze(-1).float()          # (..., G, 1)
    out = (xg * s).reshape(orig_shape)
    return out.to(scale.dtype)


def quantize_and_residual(x: Tensor, cfg: QuantConfig) -> Tuple[Tensor, Tensor, Tensor, Tuple]:
    """
    Full quantize → dequantize → residual.

    Returns
    -------
    codes    : int8 Tensor
    scale    : float Tensor
    residual : float Tensor  (x - dequantize(codes, scale))
    shape    : original shape tuple
    """
    codes, scale = quantize(x, cfg)
    x_hat = dequantize(codes, scale, x.shape, cfg)
    residual = x - x_hat
    return codes, scale, residual, x.shape


# ─────────────────────────────────────────────
# Convenience wrappers for K and V
# ─────────────────────────────────────────────

class KQuantizer:
    """
    Per-channel group quantizer for Key cache.
    K shape: (batch, heads, seq_len, head_dim)
    Groups along head_dim dimension (channel groups).
    """
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def encode(self, K: Tensor):
        # K: (B, H, T, D)  →  quantize over D
        return quantize_and_residual(K, self.cfg)

    def decode(self, codes, scale, shape) -> Tensor:
        return dequantize(codes, scale, shape, self.cfg)


class VQuantizer:
    """
    Per-token group quantizer for Value cache.
    V shape: (batch, heads, seq_len, head_dim)
    Groups along head_dim dimension.
    """
    def __init__(self, cfg: QuantConfig):
        self.cfg = cfg

    def encode(self, V: Tensor):
        return quantize_and_residual(V, self.cfg)

    def decode(self, codes, scale, shape) -> Tensor:
        return dequantize(codes, scale, shape, self.cfg)


# ─────────────────────────────────────────────
# Bit-packing helpers (INT2 / INT4)
#
# These pack int8 codes (already in range [-2^(b-1), 2^(b-1)-1]) into tight
# byte arrays for memory accounting / packed-storage mode.  Round-trip
# correctness is verified in tests.
# ─────────────────────────────────────────────

def pack_int2(codes: Tensor) -> Tensor:
    """
    Pack 1-D int8 codes in [-2, 1] into bytes (4 codes per byte, little-endian).
    Returns int8 tensor of shape (ceil(n/4),).
    """
    flat = codes.flatten().to(torch.int8)
    n = flat.numel()
    pad = (-n) % 4
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    u = (flat & 0x03).to(torch.uint8)            # 2-bit unsigned representation
    g = u.view(-1, 4)
    packed = (g[:, 0] | (g[:, 1] << 2) | (g[:, 2] << 4) | (g[:, 3] << 6)).to(torch.int8)
    return packed


def unpack_int2(packed: Tensor, numel: int) -> Tensor:
    """Inverse of pack_int2.  Returns int8 tensor of shape (numel,)."""
    u = packed.to(torch.uint8)
    out = torch.empty(u.numel() * 4, dtype=torch.uint8, device=packed.device)
    out[0::4] = u & 0x03
    out[1::4] = (u >> 2) & 0x03
    out[2::4] = (u >> 4) & 0x03
    out[3::4] = (u >> 6) & 0x03
    signed = out.to(torch.int8)
    signed[signed >= 2] -= 4                     # 2-bit signed: codes 2,3 → -2,-1
    return signed[:numel]


def pack_int4(codes: Tensor) -> Tensor:
    """
    Pack 1-D int8 codes in [-8, 7] into bytes (2 codes per byte).
    Returns int8 tensor of shape (ceil(n/2),).
    """
    flat = codes.flatten().to(torch.int8)
    n = flat.numel()
    if n % 2 != 0:
        flat = torch.cat([flat, flat.new_zeros(1)])
    u = (flat & 0x0F).to(torch.uint8)
    lo = u[0::2]
    hi = u[1::2] << 4
    packed = (lo | hi).to(torch.int8)
    return packed


def unpack_int4(packed: Tensor, numel: int) -> Tensor:
    """Inverse of pack_int4.  Returns int8 tensor of shape (numel,)."""
    u = packed.to(torch.uint8)
    lo = u & 0x0F
    hi = (u >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).flatten()[:numel]
    signed = out.to(torch.int8)
    signed[signed >= 8] -= 16
    return signed


# ─────────────────────────────────────────────
# Bit-packing helpers (INT3) — true 3-bit packing
#
# 8 signed 3-bit codes (range [-4, 3]) pack into 3 bytes (24 bits).
# ─────────────────────────────────────────────

def pack_int3(codes: Tensor) -> Tensor:
    """Pack 1-D int8 codes in [-4, 3] into bytes. 8 codes → 3 bytes."""
    flat = codes.flatten().to(torch.int8)
    n = flat.numel()
    pad = (-n) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    u = (flat & 0x07).to(torch.int64)
    g = u.view(-1, 8)
    word = (
        g[:, 0]
        | (g[:, 1] << 3)
        | (g[:, 2] << 6)
        | (g[:, 3] << 9)
        | (g[:, 4] << 12)
        | (g[:, 5] << 15)
        | (g[:, 6] << 18)
        | (g[:, 7] << 21)
    )
    b0 = (word & 0xFF)
    b1 = ((word >> 8) & 0xFF)
    b2 = ((word >> 16) & 0xFF)
    packed = torch.stack([b0, b1, b2], dim=-1).flatten().to(torch.int8)
    return packed


def unpack_int3(packed: Tensor, numel: int) -> Tensor:
    """Inverse of pack_int3.  Returns int8 tensor of shape (numel,)."""
    u = packed.to(torch.uint8).to(torch.int64)
    u = u.view(-1, 3)
    word = u[:, 0] | (u[:, 1] << 8) | (u[:, 2] << 16)
    n_groups = word.shape[0]
    out = torch.empty(n_groups, 8, dtype=torch.int64, device=packed.device)
    for i in range(8):
        out[:, i] = (word >> (i * 3)) & 0x07
    signed = out.flatten().to(torch.int8)
    signed[signed >= 4] -= 8                     # 3-bit signed: codes 4..7 → -4..-1
    return signed[:numel]


# ─────────────────────────────────────────────
# Vectorised 2-D packers for cache rows
#
# Operate on (T, D) at once and return (T, per_row_bytes).  Used by
# CAREKVCache when packed_base=True so write/read is one tensor op
# per page rather than a Python loop over tokens.
# ─────────────────────────────────────────────

def pack_int2_2d(codes: Tensor) -> Tensor:
    """(T, D) int8 in [-2, 1] → (T, D//4) int8.  D must be divisible by 4."""
    T, D = codes.shape
    assert D % 4 == 0, f"D={D} must be divisible by 4 for INT2 packing"
    u = (codes & 0x03).to(torch.uint8)
    g = u.view(T, D // 4, 4)
    packed = (g[:, :, 0] | (g[:, :, 1] << 2) | (g[:, :, 2] << 4) | (g[:, :, 3] << 6))
    return packed.to(torch.int8)


def unpack_int2_2d(packed: Tensor, D: int) -> Tensor:
    """Inverse of pack_int2_2d.  Returns (T, D) int8."""
    T = packed.shape[0]
    assert D % 4 == 0
    n_groups = D // 4
    u = packed.to(torch.uint8)
    out = torch.empty(T, n_groups, 4, dtype=torch.uint8, device=packed.device)
    out[:, :, 0] = u & 0x03
    out[:, :, 1] = (u >> 2) & 0x03
    out[:, :, 2] = (u >> 4) & 0x03
    out[:, :, 3] = (u >> 6) & 0x03
    flat = out.view(T, D).to(torch.int8)
    flat[flat >= 2] -= 4
    return flat


def pack_int4_2d(codes: Tensor) -> Tensor:
    """(T, D) int8 in [-8, 7] → (T, D//2) int8.  D must be divisible by 2."""
    T, D = codes.shape
    assert D % 2 == 0
    u = (codes & 0x0F).to(torch.uint8)
    lo = u[:, 0::2]
    hi = u[:, 1::2] << 4
    return (lo | hi).to(torch.int8)


def unpack_int4_2d(packed: Tensor, D: int) -> Tensor:
    """Inverse of pack_int4_2d.  Returns (T, D) int8."""
    T = packed.shape[0]
    u = packed.to(torch.uint8)
    lo = u & 0x0F
    hi = (u >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=-1).view(T, D).to(torch.int8)
    out[out >= 8] -= 16
    return out


def pack_int3_2d(codes: Tensor) -> Tensor:
    """(T, D) int8 in [-4, 3] → (T, D*3//8) int8.  D must be divisible by 8."""
    T, D = codes.shape
    assert D % 8 == 0, f"D={D} must be divisible by 8 for INT3 packing"
    u = (codes & 0x07).to(torch.int64)
    g = u.view(T, D // 8, 8)
    word = (
        g[:, :, 0]
        | (g[:, :, 1] << 3)
        | (g[:, :, 2] << 6)
        | (g[:, :, 3] << 9)
        | (g[:, :, 4] << 12)
        | (g[:, :, 5] << 15)
        | (g[:, :, 6] << 18)
        | (g[:, :, 7] << 21)
    )
    b0 = (word & 0xFF)
    b1 = ((word >> 8) & 0xFF)
    b2 = ((word >> 16) & 0xFF)
    packed = torch.stack([b0, b1, b2], dim=-1).flatten(1, 2).to(torch.int8)
    return packed


def unpack_int3_2d(packed: Tensor, D: int) -> Tensor:
    """Inverse of pack_int3_2d.  Returns (T, D) int8.

    Vectorized: each of the 8 nibble extractions is broadcast across an
    `arange(8)*3` shift vector instead of the original Python for-loop.
    Result is bit-identical to the iterative version (verified in tests).
    """
    T = packed.shape[0]
    assert D % 8 == 0
    n_groups = D // 8
    u = packed.to(torch.uint8).view(T, n_groups, 3).to(torch.int64)
    word = u[:, :, 0] | (u[:, :, 1] << 8) | (u[:, :, 2] << 16)   # (T, n_groups) uint24-in-int64
    shifts = torch.arange(8, device=word.device, dtype=torch.int64) * 3
    out = (word.unsqueeze(-1) >> shifts) & 0x07                  # (T, n_groups, 8)
    flat = out.view(T, D).to(torch.int8)
    flat[flat >= 4] -= 8
    return flat


# ─────────────────────────────────────────────
# Generic dispatchers used by CAREKVCache.
# ─────────────────────────────────────────────

def packed_row_bytes(D: int, bits: int) -> int:
    """Bytes needed for one packed row of D codes at `bits` bits."""
    if bits == 2:
        assert D % 4 == 0
        return D // 4
    if bits == 3:
        assert D % 8 == 0
        return D * 3 // 8
    if bits == 4:
        assert D % 2 == 0
        return D // 2
    raise ValueError(f"Unsupported base_bits={bits} for packed storage")


def pack_codes_2d(codes: Tensor, bits: int) -> Tensor:
    """(T, D) int8 → (T, packed_row_bytes(D, bits)) int8."""
    if bits == 2:
        return pack_int2_2d(codes)
    if bits == 3:
        return pack_int3_2d(codes)
    if bits == 4:
        return pack_int4_2d(codes)
    raise ValueError(f"Unsupported base_bits={bits}")


def unpack_codes_2d(packed: Tensor, bits: int, D: int) -> Tensor:
    """Inverse of pack_codes_2d.  Returns (T, D) int8."""
    if bits == 2:
        return unpack_int2_2d(packed, D)
    if bits == 3:
        return unpack_int3_2d(packed, D)
    if bits == 4:
        return unpack_int4_2d(packed, D)
    raise ValueError(f"Unsupported base_bits={bits}")
