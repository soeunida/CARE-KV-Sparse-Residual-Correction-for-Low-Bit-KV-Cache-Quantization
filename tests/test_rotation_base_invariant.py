"""tests/test_rotation_base_invariant.py

Invariant tests for the rotation base quantizers (arms 4/5 of the rotation+
CARE-KV stack). Helper-level, CPU-only, fast — no model load.

  T1  orthonormality           R @ R.T == I  (Hadamard + random)
  T2  lossless rotation        (K @ R) @ R.T == K
  T3  R = I reduction          rotate-quant with R=I == plain per-channel/token quant
  T4  determinism              seeded random R bit-identical across cache clears
  T5  dispatch                 dispatch_base_kv_quant("randrot_style", ...) works
"""
import torch

from CARE_KV.care_kv.kivi_helpers import (
    _get_hadamard, _get_randrot, _random_orthonormal,
    _rotate_quant_per_channel, _rotate_quant_per_token,
    quant_dequant_kivi_k, quant_dequant_kivi_v,
    quant_dequant_randrot_k, quant_dequant_randrot_v,
    dispatch_base_kv_quant, RANDROT_SEED,
)
import CARE_KV.care_kv.kivi_helpers as kh

D = 64
T = 16


def _K():
    g = torch.Generator().manual_seed(0)
    return torch.randn(2, T, D, generator=g)


def test_T1_orthonormal():
    eye = torch.eye(D)
    H = _get_hadamard(D, "cpu", torch.float32)
    R = _get_randrot(D, "cpu", torch.float32)
    assert torch.allclose(H @ H.T, eye, atol=1e-5)
    assert torch.allclose(R @ R.T, eye, atol=1e-5)


def test_T2_lossless_rotation_roundtrip():
    K = _K().to(torch.float32)
    for R in (_get_hadamard(D, "cpu", torch.float32),
              _get_randrot(D, "cpu", torch.float32)):
        assert torch.allclose((K @ R) @ R.T, K, atol=1e-4)


def test_T3_R_eq_I_reduces_to_per_channel_quant():
    K = _K()
    V = _K()
    eye = torch.eye(D)
    for bits in (2, 3, 4):
        assert torch.allclose(
            _rotate_quant_per_channel(K, bits, eye),
            quant_dequant_kivi_k(K, bits), atol=1e-6), f"K bits={bits}"
        assert torch.allclose(
            _rotate_quant_per_token(V, bits, eye),
            quant_dequant_kivi_v(V, bits), atol=1e-6), f"V bits={bits}"


def test_T4_randrot_deterministic_across_cache_clear():
    R1 = _get_randrot(D, "cpu", torch.float32).clone()
    kh._RANDROT_CACHE.clear()
    R2 = _get_randrot(D, "cpu", torch.float32)
    assert torch.equal(R1, R2)
    assert torch.equal(_random_orthonormal(D, RANDROT_SEED, "cpu"),
                       _random_orthonormal(D, RANDROT_SEED, "cpu"))


def test_T5_dispatch_randrot_shapes_and_finite():
    K, V = _K(), _K()
    Kh, Vh = dispatch_base_kv_quant("randrot_style", K, V, 3, 3)
    assert Kh.shape == K.shape and Vh.shape == V.shape
    assert torch.isfinite(Kh).all() and torch.isfinite(Vh).all()
    e2 = (quant_dequant_randrot_k(K, 2) - K).pow(2).mean()
    e4 = (quant_dequant_randrot_k(K, 4) - K).pow(2).mean()
    assert e4 < e2
