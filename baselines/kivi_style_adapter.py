"""baselines/kivi_style_adapter.py — same-condition KIVI-style reimplementation.

KIVI's distinguishing trait is **asymmetric** KV quantization:
  - K is quantized **per-channel** (one scale per channel, taken across all
    tokens in the sequence).
  - V is quantized **per-token**  (one scale per token, taken across all
    channels in the head_dim).

This adapter implements that round-trip *in-place* on K and V at the
attention layer, then runs attention against (K_hat, V_hat). It is a
**same-condition reimplementation** — same model, same forward path,
same PPL eval — and is NOT the official KIVI repo (which ships custom
CUDA kernels).

Implementation: monkey-patch every `LlamaAttention.forward` in the model
to apply quantize-dequantize to K (post-RoPE) and V before the attention
matmul.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
from torch import Tensor

from transformers import LlamaForCausalLM
from .common import KVMethodAdapter, DEVICE, fp16_kv_mb


def _quant_dequant_per_channel_K(K: Tensor, bits: int) -> Tensor:
    """K shape: (B, Hkv, T, D). Returns K_hat with the same shape.

    KIVI per-channel K: one scale per (B, Hkv, channel), taken as max-abs
    across the time dimension.
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    # scale shape (B, Hkv, 1, D)
    scale = K.abs().amax(dim=2, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    codes = (K / scale).round().clamp(qmin, qmax)
    return codes * scale


def _quant_dequant_per_token_V(V: Tensor, bits: int) -> Tensor:
    """V shape: (B, Hkv, T, D). Returns V_hat with the same shape.

    KIVI per-token V: one scale per (B, Hkv, token), taken as max-abs
    across the head_dim.
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    scale = V.abs().amax(dim=3, keepdim=True) / qmax
    scale = scale.clamp(min=1e-8)
    codes = (V / scale).round().clamp(qmin, qmax)
    return codes * scale


def patch_kivi_style(model, bits_k: int, bits_v: int):
    """Monkey-patch every Llama attention layer to quantize K (post-RoPE)
    per-channel and V per-token before the attention matmul.

    The implementation wraps `LlamaAttention.forward`. We compute Q/K/V
    + RoPE via the original forward up to (but not including) the
    attention computation, then quantize-dequantize K and V, then run
    the rest. The cleanest hook point is to wrap the whole `forward` and
    let the original do the proj + RoPE, then patch K/V before the
    attention math.

    BUT modern HF LlamaAttention.forward is one monolithic function,
    so we instead monkey-patch at the `apply_rotary_pos_emb` boundary:
    after the apply_rotary_pos_emb call inside forward, wrap K/V
    through quant-dequant.

    Implementation strategy here: replace the entire
    `LlamaAttention.forward` with a wrapper that imitates the original
    but inserts the K/V quant-dequant step. Because this is HF-version
    dependent and risky, we use a *function decorator on apply_rotary_pos_emb*
    that wraps the K return value.
    """
    from transformers.models.llama import modeling_llama

    # Capture the original apply_rotary_pos_emb
    orig_apply_rotary = modeling_llama.apply_rotary_pos_emb

    def wrapped_apply_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_out, k_out = orig_apply_rotary(q, k, cos, sin,
                                            position_ids=position_ids,
                                            unsqueeze_dim=unsqueeze_dim)
        # k_out shape after rotary: (B, Hkv, T, D)
        k_hat = _quant_dequant_per_channel_K(k_out, bits_k)
        return q_out, k_hat

    # Patch the function symbol in the module that LlamaAttention reads from.
    modeling_llama.apply_rotary_pos_emb = wrapped_apply_rotary

    # For V: monkey-patch each module's v_proj to wrap its output.
    # We walk model and find every LlamaAttention.
    def v_proj_wrap(v_proj_module):
        orig_forward = v_proj_module.forward
        def new_forward(x):
            v_lin = orig_forward(x)
            # v_lin shape: (B, T, Hkv*D)
            # Reshape, quantize per token, reshape back.
            B, T, HD = v_lin.shape
            cfg = model.config
            hkv = cfg.num_key_value_heads
            d = cfg.hidden_size // cfg.num_attention_heads
            v = v_lin.view(B, T, hkv, d).transpose(1, 2)   # (B, Hkv, T, D)
            v_hat = _quant_dequant_per_token_V(v, bits_v)
            return v_hat.transpose(1, 2).reshape(B, T, HD)
        v_proj_module.forward = new_forward

    for name, module in model.named_modules():
        if hasattr(module, "v_proj") and hasattr(module.v_proj, "forward"):
            v_proj_wrap(module.v_proj)

    # Return the function the caller should call to UNDO the patch (for
    # safety in multi-cell runs that re-instantiate the model).
    def unpatch():
        modeling_llama.apply_rotary_pos_emb = orig_apply_rotary
    return unpatch


class KIVIStyleAdapter(KVMethodAdapter):
    family = "kivi_style"
    is_official = False
    is_reimplementation = True
    uses_residual = False
    uses_mixed_precision = False  # KIVI itself is uniform low-bit
    uses_token_eviction = False
    uses_query_aware_routing = False

    def __init__(self, bits_k: int = 3, bits_v: int = 3):
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.name = f"KIVI_style_INT{bits_k}K_INT{bits_v}V"
        self.bit_width = f"K=INT{bits_k}, V=INT{bits_v}"
        self.k_quant_scheme = f"per-channel (scale across tokens), INT{bits_k} symmetric"
        self.v_quant_scheme = f"per-token  (scale across channels), INT{bits_v} symmetric"
        self._unpatch = None

    def setup_model(self, model_id: str):
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map=DEVICE if DEVICE == "cuda" else None,
        )
        m.config.use_cache = False
        m.eval()
        self._unpatch = patch_kivi_style(m, self.bits_k, self.bits_v)
        return m

    def teardown(self):
        if self._unpatch is not None:
            self._unpatch()
            self._unpatch = None

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64):
        # KIVI's K: bits_k per element + 1 fp16 scale per (kv-head, channel)
        # K bytes per layer: T*D * bits_k/8 + 2*hkv*D (scale)
        # KIVI's V: bits_v per element + 1 fp16 scale per (kv-head, token)
        # V bytes per layer: T*D * bits_v/8 + 2*hkv*T  (scale)
        k_bytes_per_layer = (seq_len * hkv * head_dim * self.bits_k / 8.0
                              + 2 * hkv * head_dim)
        v_bytes_per_layer = (seq_len * hkv * head_dim * self.bits_v / 8.0
                              + 2 * hkv * seq_len)
        total = num_layers * (k_bytes_per_layer + v_bytes_per_layer)
        total_mb = total / (1024 * 1024)
        fp16 = fp16_kv_mb(seq_len, num_layers, hkv, head_dim)
        return dict(estimated_kv_memory_MB=round(total_mb, 4),
                    estimated_total_cache_memory_MB=round(total_mb, 4),
                    vs_fp16_kv_memory_ratio=round(total_mb / max(fp16, 1e-9), 4))

    def notes(self) -> str:
        return (f"KIVI-style same-condition reimplementation. Per-channel K "
                f"(INT{self.bits_k}) + per-token V (INT{self.bits_v}). "
                f"Implemented by monkey-patching apply_rotary_pos_emb to "
                f"quant-dequant K post-RoPE, and v_proj to quant-dequant V. "
                f"NOT the official KIVI repo (no custom CUDA kernels).")
