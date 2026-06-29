"""baselines/rotatekv_style.py — RotateKV-style same-condition reimplementation.

The idea (from the RotateKV / rotation-based quantization line of work):
apply a fixed, deterministic orthonormal rotation `H` along the
head_dim axis before quantization, then apply the inverse `H^T` after
dequantization. Because `H @ H^T = I`, the round trip is mathematically
identity in fp; the win comes from quantization being friendlier to a
rotated distribution (Hadamard rotation spreads outliers across all
channels, reducing per-channel scale variance).

Implementation:
 - We use a **Walsh-Hadamard matrix** of size `head_dim`, normalized
   to be orthonormal (H / sqrt(D)). The Sylvester construction is
   pure-PyTorch (no scipy dep) and only needs `head_dim` to be a
   power of 2 — which it is for every Llama-family head_dim we care
   about (64, 128, 256, ...).
 - The patch is applied **post-RoPE** so it composes with CARE-KV's
   post-RoPE cache. For K: rotate post-RoPE K, quantize per-channel,
   dequantize, inverse-rotate. For V: rotate, quantize per-token,
   dequantize, inverse-rotate.

Honest framing:
 - "RotateKV-style same-condition reimplementation" — fixed Hadamard
   rotation + per-channel K / per-token V uniform INT3 quant.
 - NOT the official RotateKV (which uses learned rotations + may use
   different quant granularity). This adapter freezes the rotation as
   a Walsh-Hadamard matrix and uses uniform INT quant.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from transformers import LlamaForCausalLM
from .common import KVMethodAdapter, DEVICE, fp16_kv_mb


def _walsh_hadamard(n: int, device, dtype=torch.float32) -> Tensor:
    """Construct an n×n Walsh-Hadamard matrix via Sylvester recursion.

    Requires `n` to be a power of 2. Returns an *orthonormal* matrix
    H / sqrt(n), so H @ H.T == I (within fp tolerance).
    """
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Walsh-Hadamard requires n=2^k; got n={n}")
    H = torch.ones(1, 1, device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    # Normalize to orthonormal
    return H / (n ** 0.5)


# Cache the Hadamard matrix per (device, dtype, n) so each forward
# doesn't re-build it.
_HADAMARD_CACHE: dict = {}

def _get_hadamard(n: int, device, dtype) -> Tensor:
    key = (n, str(device), dtype)
    if key not in _HADAMARD_CACHE:
        _HADAMARD_CACHE[key] = _walsh_hadamard(n, device, dtype)
    return _HADAMARD_CACHE[key]


def _rotate_quant_dequant_unrotate_K(K: Tensor, bits: int) -> Tensor:
    """K shape: (B, Hkv, T, D). Rotate K @ H along D, per-channel quant,
    dequant, inverse-rotate. H is orthonormal so the fp round-trip is
    identity; the quant noise lives in the rotated basis.
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    H = _get_hadamard(K.shape[-1], K.device, torch.float32)        # (D, D)
    K32 = K.to(torch.float32)
    K_rot = K32 @ H                                                 # rotate along last dim
    scale = K_rot.abs().amax(dim=-2, keepdim=True) / qmax           # per-channel across T
    scale = scale.clamp(min=1e-8)
    codes = (K_rot / scale).round().clamp(qmin, qmax)
    K_rot_hat = codes * scale
    K_hat = K_rot_hat @ H.T                                         # inverse rotate (orthonormal)
    return K_hat.to(K.dtype)


def _rotate_quant_dequant_unrotate_V(V: Tensor, bits: int) -> Tensor:
    """V shape: (B, Hkv, T, D). Rotate V @ H along D, per-token quant,
    dequant, inverse-rotate.
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    H = _get_hadamard(V.shape[-1], V.device, torch.float32)
    V32 = V.to(torch.float32)
    V_rot = V32 @ H
    scale = V_rot.abs().amax(dim=-1, keepdim=True) / qmax            # per-token across rotated D
    scale = scale.clamp(min=1e-8)
    codes = (V_rot / scale).round().clamp(qmin, qmax)
    V_rot_hat = codes * scale
    V_hat = V_rot_hat @ H.T
    return V_hat.to(V.dtype)


def patch_rotatekv_style(model, bits_k: int, bits_v: int):
    """Monkey-patch apply_rotary_pos_emb to add the rotate+quant+unrotate
    step on post-RoPE K, and wrap v_proj for the same on V.
    Returns an `unpatch` callable that restores the rotary symbol.
    """
    from transformers.models.llama import modeling_llama

    orig_apply_rotary = modeling_llama.apply_rotary_pos_emb

    def wrapped_apply_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_out, k_out = orig_apply_rotary(q, k, cos, sin,
                                          position_ids=position_ids,
                                          unsqueeze_dim=unsqueeze_dim)
        # k_out: (B, Hkv, T, D)
        k_hat = _rotate_quant_dequant_unrotate_K(k_out, bits_k)
        return q_out, k_hat
    modeling_llama.apply_rotary_pos_emb = wrapped_apply_rotary

    cfg_model = model.config
    hkv = cfg_model.num_key_value_heads
    d = cfg_model.hidden_size // cfg_model.num_attention_heads

    def v_proj_wrap(v_proj_module):
        orig_forward = v_proj_module.forward
        def new_forward(x):
            v_lin = orig_forward(x)
            B, T, HD = v_lin.shape
            v = v_lin.view(B, T, hkv, d).transpose(1, 2)
            v_hat = _rotate_quant_dequant_unrotate_V(v, bits_v)
            return v_hat.transpose(1, 2).reshape(B, T, HD)
        v_proj_module.forward = new_forward

    for name, module in model.named_modules():
        if hasattr(module, "v_proj") and hasattr(module.v_proj, "forward"):
            v_proj_wrap(module.v_proj)

    def unpatch():
        modeling_llama.apply_rotary_pos_emb = orig_apply_rotary
    return unpatch


class RotateKVStyleAdapter(KVMethodAdapter):
    family = "rotatekv_style"
    is_official = False
    is_reimplementation = True
    is_unsupported = False
    uses_residual = False
    uses_mixed_precision = False
    uses_token_eviction = False
    uses_query_aware_routing = False

    def __init__(self, bits_k: int = 3, bits_v: int = 3):
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.name = f"RotateKV_style_INT{bits_k}K_INT{bits_v}V"
        self.bit_width = f"K=INT{bits_k}, V=INT{bits_v}"
        self.k_quant_scheme = (f"Walsh-Hadamard rotation + per-channel "
                                f"INT{bits_k} symmetric + inverse rotation")
        self.v_quant_scheme = (f"Walsh-Hadamard rotation + per-token "
                                f"INT{bits_v} symmetric + inverse rotation")
        self._unpatch = None

    def setup_model(self, model_id: str):
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map=DEVICE if DEVICE == "cuda" else None,
        )
        m.config.use_cache = False
        m.eval()
        self._unpatch = patch_rotatekv_style(m, self.bits_k, self.bits_v)
        return m

    def teardown(self):
        if self._unpatch is not None:
            self._unpatch()
            self._unpatch = None

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                          hkv: int = 4, head_dim: int = 64):
        # Same theoretical footprint as KIVI: per-channel K + per-token V + fp16 scale.
        # The Hadamard matrix is a fixed constant — not counted as KV memory.
        k_bytes = (seq_len * hkv * head_dim * self.bits_k / 8.0
                    + 2 * hkv * head_dim)
        v_bytes = (seq_len * hkv * head_dim * self.bits_v / 8.0
                    + 2 * hkv * seq_len)
        total_mb = num_layers * (k_bytes + v_bytes) / (1024 * 1024)
        fp16 = fp16_kv_mb(seq_len, num_layers, hkv, head_dim)
        return dict(
            estimated_kv_memory_MB=round(total_mb, 4),
            estimated_total_cache_memory_MB=round(total_mb, 4),
            vs_fp16_kv_memory_ratio=round(total_mb / max(fp16, 1e-9), 4),
            base_memory_MB=round(total_mb, 4),
            residual_memory_MB=0.0,
            base_quantizer="rotatekv_style",
        )

    def notes(self) -> str:
        return (f"RotateKV-style same-condition reimplementation. Fixed "
                f"Walsh-Hadamard rotation along head_dim (orthonormal "
                f"H/sqrt(D)) + per-channel K INT{self.bits_k} + per-token "
                f"V INT{self.bits_v} + inverse rotation. NOT the official "
                f"RotateKV repo (no learned rotations, no calibration).")
