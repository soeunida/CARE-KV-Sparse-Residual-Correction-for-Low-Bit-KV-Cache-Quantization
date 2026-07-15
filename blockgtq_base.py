"""care_kv/blockgtq_base.py — Block-GTQ as a CARE-KV base quantizer.

Bridges the external Block-GTQ repo (RoPE-aware per-(layer, KV-head) bit
allocation for the K cache; TurboQuant-MSE for V) into CARE-KV's
base-quantizer dispatch (see kivi_helpers.dispatch_base_kv_quant).

Unlike the data-free base quantizers (kivi_style / rotatekv_style /
randrot_style), Block-GTQ is **calibrated per (layer, KV-head)**: its
allocator distributes an average bit budget non-uniformly across RoPE
frequency blocks from a Q/K energy score. So this module keeps a global
registry of calibrated quantizers, populated once per model by
`calibrate(...)` BEFORE the CARE-KV patch runs, and consumed at store time
by `kv_hat(layer_id, kv_head, K, V)`.

Calibration is done **post-RoPE** (`post_rope=True`) to match CARE-KV's
post-RoPE K cache: the stored K the residual is computed against is
post-RoPE, so the base reconstruction K_hat must live in the same
coordinate system. The V side uses Block-GTQ's TurboQuant-MSE at v_bits.

Usage (driver):
    import CARE_KV.care_kv.blockgtq_base as bgb
    bgb.calibrate(model, calib_ids, k_avg_bits=3.0, v_bits=3, device="cuda:0")
    # ... then patch_llama_model(model, cfg) with cfg.base_quantizer="blockgtq_style"
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import torch
from torch import Tensor

# Make the external Block-GTQ package importable. Overridable via env.
_BLOCKGTQ_PATH = os.environ.get("BLOCKGTQ_PATH", "/home/soeun/blockgtq")
if _BLOCKGTQ_PATH not in sys.path:
    sys.path.insert(0, _BLOCKGTQ_PATH)

# Registry: calibrated per-(layer, KV-head) quantizers.
#   _REG["kq"][li][hi] : blockgtq.block_gtq_pipeline.BlockGTQPipeline (K)
#   _REG["vq"][li][hi] : blockgtq.tq.TurboQuantMSE                    (V)
_REG: dict = {"kq": None, "vq": None, "meta": None}


def is_calibrated() -> bool:
    return _REG["kq"] is not None and _REG["vq"] is not None


def reset() -> None:
    _REG["kq"] = None
    _REG["vq"] = None
    _REG["meta"] = None


@torch.no_grad()
def calibrate(model, calib_ids: Tensor, k_avg_bits: float, v_bits: int,
              device, n_calib_tokens: int = 2048) -> None:
    """Calibrate Block-GTQ per (layer, KV-head) on `calib_ids` and populate
    the registry.

    Args:
        model: the RAW (un-patched) HF Llama-style model. Must be called
            BEFORE patch_llama_model, because calibration hooks the original
            q_proj / k_proj forwards.
        calib_ids: (1, T) input ids; the first n_calib_tokens are used.
        k_avg_bits: average K bit budget (allocator distributes non-uniformly).
        v_bits: uniform V bit width (TurboQuant-MSE).
        device: target device.
        n_calib_tokens: number of calibration tokens.
    """
    from blockgtq.calibration import collect_qk_activations
    from blockgtq.unaccelerated import build_unaccelerated_quantizers

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    nkv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = cfg.hidden_size // cfg.num_attention_heads

    # Post-RoPE calibration → matches CARE-KV's post-RoPE K cache.
    layer_data = collect_qk_activations(
        model, calib_ids, device, n_calib_tokens=n_calib_tokens,
        post_rope=True,
    )
    kq, vq = build_unaccelerated_quantizers(
        layer_data, n_layers, nkv, hd,
        k_avg_bits=float(k_avg_bits), v_bits=int(v_bits), device=device,
    )
    _REG["kq"] = kq
    _REG["vq"] = vq
    _REG["meta"] = dict(n_layers=n_layers, nkv=nkv, hd=hd,
                        k_avg_bits=float(k_avg_bits), v_bits=int(v_bits),
                        n_calib_tokens=int(n_calib_tokens))


@torch.no_grad()
def kv_hat(layer_id: int, kv_head: int, K: Tensor, V: Tensor
           ) -> Tuple[Tensor, Tensor]:
    """Return (K_hat, V_hat) — the Block-GTQ reconstruction of the
    post-RoPE K and the TurboQuant-MSE reconstruction of V for one
    (layer, KV-head). Shapes preserved: (..., T, hd) in → (..., T, hd) out.
    """
    if not is_calibrated():
        raise RuntimeError(
            "blockgtq_base.kv_hat called before calibrate(); the "
            "blockgtq_style base quantizer needs per-(layer,head) "
            "calibration. Call blockgtq_base.calibrate(model, ...) first.")
    kq = _REG["kq"][layer_id][kv_head]
    vq = _REG["vq"][layer_id][kv_head]
    hd = K.shape[-1]
    # The calibrated quantizers live on the calibration device (cuda:0). For a
    # device_map="auto" (sharded) model, K/V for later layers may be on another
    # GPU — run the quant on the quantizer's device, then move back. For single-
    # GPU models pdev == K.device, so these .to() calls are no-ops.
    pdev = getattr(kq, "device", None) or torch.device("cuda:0")
    K2 = K.reshape(-1, hd).float().to(pdev).contiguous()
    V2 = V.reshape(-1, hd).float().to(pdev).contiguous()
    K_hat = kq.compress_decompress(K2).reshape(K.shape)
    V_hat = vq.compress_decompress(V2).reshape(V.shape)
    # The float32 reconstruction can slightly overshoot the input magnitude
    # (quantization error + TQ-MSE norm rescaling). For massive-activation models
    # (e.g. Yi-34B, |K|>3e4) that overshoot exceeds the fp16 max (65504), so the
    # cast to fp16 produces +inf → NaN in attention. Saturate to the fp16 finite
    # range before casting (no-op for well-scaled models).
    if os.environ.get("CAREKV_NAN_DEBUG") == "1":
        if not (torch.isfinite(K_hat).all() and torch.isfinite(V_hat).all()):
            print(f"[nandbg] kv_hat NONFINITE L{layer_id}H{kv_head}: "
                  f"|K|max={K.abs().max().item():.0f} -> K_hat_finite="
                  f"{torch.isfinite(K_hat).all().item()} |K_hat|max="
                  f"{K_hat[torch.isfinite(K_hat)].abs().max().item() if torch.isfinite(K_hat).any() else float('nan'):.0f} "
                  f"V_hat_finite={torch.isfinite(V_hat).all().item()}", flush=True)
    if K.dtype == torch.float16:
        K_hat = K_hat.clamp_(-65504.0, 65504.0)
        V_hat = V_hat.clamp_(-65504.0, 65504.0)
    return K_hat.to(K.device, K.dtype), V_hat.to(V.device, V.dtype)
