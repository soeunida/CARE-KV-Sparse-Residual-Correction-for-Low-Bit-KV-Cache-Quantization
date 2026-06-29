"""care_kv/kivi_helpers.py — pure quant-dequant helpers for CARE-KV's
base-quantizer dispatch.

Lives in the core package (not under baselines/) so that layer.py can
import it without triggering baselines/__init__.py's adapter imports
(which would create a circular import back into care_kv).

Quantizers supported by the dispatch:
  - "kivi_style"     : per-channel K (scale across T) + per-token V (scale across D)
  - "rotatekv_style" : Walsh-Hadamard rotate, then per-channel K /
                        per-token V quant, then inverse rotate
  - "kvquant_style"  : same per-channel K + per-token V; the pre-RoPE
                        K-store-mode choice is enforced in the adapter,
                        not here (residual-side cache is post-RoPE either way)

All helpers are shape-agnostic — they take `(..., T, D)` and return the
same shape. None of them use CUDA kernels.

Used by:
  - layer.py:_write_pages_for_kv_head / decode-mode append, when
    cfg.base_quantizer != "uniform".
  - baselines/kivi_style_quantizer.py KIVIStyleQuantizer (re-exports
    the kivi pair).
"""
from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor


# ─────────────────────────────────────────────
# KIVI-style: per-channel K + per-token V
# ─────────────────────────────────────────────

def quant_dequant_kivi_k(K: Tensor, bits: int) -> Tensor:
    """K shape: (..., T, D). Symmetric INT quantization with **per-channel**
    scale (computed across the T axis).
    Returns a tensor with the same shape and dtype as K, holding the
    dequantized K_hat = round(K/scale)·scale.
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    scale = K.abs().amax(dim=-2, keepdim=True) / qmax    # (..., 1, D)
    scale = scale.clamp(min=1e-8)
    codes = (K / scale).round().clamp(qmin, qmax)
    return codes * scale


def quant_dequant_kivi_v(V: Tensor, bits: int) -> Tensor:
    """V shape: (..., T, D). Symmetric INT quantization with **per-token**
    scale (computed across the D axis).
    Returns a tensor with the same shape and dtype as V.
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    scale = V.abs().amax(dim=-1, keepdim=True) / qmax    # (..., T, 1)
    scale = scale.clamp(min=1e-8)
    codes = (V / scale).round().clamp(qmin, qmax)
    return codes * scale


# ─────────────────────────────────────────────
# Orthonormal rotations: Walsh-Hadamard (fixed) and random-Gaussian (seeded)
#
# Both rotate channels (head_dim) before quantization; the inverse R.T is
# applied after dequant (R @ R.T = I → lossless in fp). rotatekv_style uses the
# fixed Walsh-Hadamard matrix; randrot_style (TurboQuant-style) uses a SEEDED
# random orthonormal matrix — seeded so the SAME R is reconstructed at store and
# decode time.
# ─────────────────────────────────────────────

_HADAMARD_CACHE: dict = {}
_RANDROT_CACHE: dict = {}
RANDROT_SEED = 42   # fixed; store-time and decode-time must agree on R

def _walsh_hadamard(n: int, device, dtype=torch.float32) -> Tensor:
    """Sylvester construction of an n×n Walsh-Hadamard matrix, normalized
    to be orthonormal (H/sqrt(n) so H @ H.T = I within fp tolerance).
    Requires n to be a power of 2.
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Walsh-Hadamard requires n=2^k; got n={n}")
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    return H / (n ** 0.5)


def _get_hadamard(n: int, device, dtype) -> Tensor:
    key = (n, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[key] = _walsh_hadamard(n, device, dtype)
    return _HADAMARD_CACHE[key]


def _random_orthonormal(n: int, seed: int, device, dtype=torch.float32) -> Tensor:
    """Deterministic random orthonormal n×n matrix via QR of a seeded Gaussian
    (sign-fixed diagonal) so it is bit-identical across processes/devices."""
    gen = torch.Generator(device="cpu").manual_seed(seed + n)
    A = torch.randn(n, n, generator=gen, dtype=torch.float32)
    Q, Rm = torch.linalg.qr(A)
    d = torch.sign(torch.diagonal(Rm))
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


def _get_randrot(n: int, device, dtype, seed: int = RANDROT_SEED) -> Tensor:
    key = (n, str(device), dtype, seed)
    if key not in _RANDROT_CACHE:
        _RANDROT_CACHE[key] = _random_orthonormal(n, seed, device, dtype)
    return _RANDROT_CACHE[key]


def _rotate_quant_per_channel(K: Tensor, bits: int, R: Tensor) -> Tensor:
    """K (..., T, D): rotate K @ R, per-channel (across T) symmetric INT quant,
    dequant, inverse rotate @ R.T. R=I reduces to plain per-channel quant."""
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    K_rot = K.to(torch.float32) @ R
    scale = (K_rot.abs().amax(dim=-2, keepdim=True) / qmax).clamp(min=1e-8)
    codes = (K_rot / scale).round().clamp(qmin, qmax)
    return ((codes * scale) @ R.T).to(K.dtype)


def _rotate_quant_per_token(V: Tensor, bits: int, R: Tensor) -> Tensor:
    """V (..., T, D): rotate V @ R, per-token (across D) symmetric INT quant,
    dequant, inverse rotate @ R.T."""
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    V_rot = V.to(torch.float32) @ R
    scale = (V_rot.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
    codes = (V_rot / scale).round().clamp(qmin, qmax)
    return ((codes * scale) @ R.T).to(V.dtype)


def quant_dequant_rotatekv_k(K: Tensor, bits: int) -> Tensor:
    """Walsh-Hadamard rotate + per-channel quant."""
    return _rotate_quant_per_channel(
        K, bits, _get_hadamard(K.shape[-1], K.device, torch.float32))


def quant_dequant_rotatekv_v(V: Tensor, bits: int) -> Tensor:
    """Walsh-Hadamard rotate + per-token quant."""
    return _rotate_quant_per_token(
        V, bits, _get_hadamard(V.shape[-1], V.device, torch.float32))


def quant_dequant_randrot_k(K: Tensor, bits: int) -> Tensor:
    """Seeded random-orthonormal rotate + per-channel quant (TurboQuant-style)."""
    return _rotate_quant_per_channel(
        K, bits, _get_randrot(K.shape[-1], K.device, torch.float32))


def quant_dequant_randrot_v(V: Tensor, bits: int) -> Tensor:
    """Seeded random-orthonormal rotate + per-token quant."""
    return _rotate_quant_per_token(
        V, bits, _get_randrot(V.shape[-1], V.device, torch.float32))


# ─────────────────────────────────────────────
# Dispatch helper used by layer.py
# ─────────────────────────────────────────────

def dispatch_base_kv_quant(
    name: str, K: Tensor, V: Tensor, k_bits: int, v_bits: int,
) -> Tuple[Tensor, Tensor]:
    """Return (K_hat, V_hat) for the given base-quantizer name on
    `(..., T, D)` inputs. The CARE-KV cache stores post-RoPE K, so
    `name="kvquant_style"` is treated as the post-RoPE variant here
    (the true pre-RoPE KVQuant is incompatible with the cache layout
    and is blocked in the adapter — see baselines/kvquant_style.py).
    """
    if name == "kivi_style":
        return quant_dequant_kivi_k(K, k_bits), quant_dequant_kivi_v(V, v_bits)
    if name == "rotatekv_style":
        return quant_dequant_rotatekv_k(K, k_bits), quant_dequant_rotatekv_v(V, v_bits)
    if name == "randrot_style":
        return quant_dequant_randrot_k(K, k_bits), quant_dequant_randrot_v(V, v_bits)
    if name == "kvquant_style":
        # Post-RoPE variant — same shape as KIVI's per-channel K.
        return quant_dequant_kivi_k(K, k_bits), quant_dequant_kivi_v(V, v_bits)
    raise ValueError(
        f"dispatch_base_kv_quant: unsupported base_quantizer={name!r}. "
        f"Known: kivi_style, rotatekv_style, randrot_style, kvquant_style"
    )
