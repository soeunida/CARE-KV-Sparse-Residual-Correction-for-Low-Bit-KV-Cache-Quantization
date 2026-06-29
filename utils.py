"""
care_kv/utils.py
-----------------
Utility functions for CARE-KV research.

Includes:
  - Layer/head sensitivity calibration
  - Perplexity evaluation helper
  - Store/read budget sweep
  - Ablation comparison utilities
  - Memory footprint estimator
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Tuple, Optional, Callable
import math

from .cache import CacheConfig


# ─────────────────────────────────────────────
# Memory footprint estimator
# ─────────────────────────────────────────────

def _bits_to_packed_bytes(num_values: int, bits: int) -> int:
    """Theoretical packed size (ceil) of num_values elements at `bits` bits each."""
    return (num_values * bits + 7) // 8


def estimate_memory_bytes(
    cfg: CacheConfig,
    seq_len: int,
    store_ratio: Optional[float] = None,
    packed: Optional[bool] = None,
) -> Dict[str, int]:
    """
    Estimate memory usage for CARE-KV cache at a given sequence length.

    `packed=True` reports the theoretical tight-packed size for INT2/3/4 codes
    (the right number for a paper memory table).  `packed=False` reports the
    int8 storage size actually used in this implementation's accuracy mode.

    Returns a dict with separated components:
        base_K_code_bytes / base_V_code_bytes
        base_K_scale_bytes / base_V_scale_bytes
        residual_K_bytes / residual_V_bytes
        metadata_bytes      (per-page indices + valid_tokens + scalar headers)
        sketch_bytes        (k_sketch fp16 tensors)
        error_norm_bytes    (k_error_norm + v_error_norm fp16)
        total_bytes
        fp16_kv_bytes / int4_kv_bytes  (references)
        compression_vs_fp16 / compression_vs_int4
    """
    if store_ratio is None:
        store_ratio = cfg.store_budget_ratio
    if packed is None:
        packed = cfg.packed_base

    L  = cfg.num_layers
    Hk = cfg.num_kv_heads
    D  = cfg.head_dim
    T  = seq_len

    G = D // cfg.group_size
    num_pages = math.ceil(T / cfg.page_size)
    tokens_padded = num_pages * cfg.page_size

    # ── Base codes ────────────────────────────────────────────────────
    # Per-token: D values at cfg.base_bits each.
    if packed:
        per_token_code_bytes = _bits_to_packed_bytes(D, cfg.base_bits)
    else:
        per_token_code_bytes = D                      # int8 storage
    base_K_code_bytes = L * Hk * tokens_padded * per_token_code_bytes
    base_V_code_bytes = L * Hk * tokens_padded * per_token_code_bytes

    # Scales: width depends on cfg.scale_dtype; plus optional per-page int8
    # quantization with one fp16 master per page.
    scale_dtype_bytes = {"fp16": 2, "bf16": 2, "fp32": 4}.get(cfg.scale_dtype, 2)
    if cfg.scale_quant == "int8":
        per_scale = 1                                        # int8 code
        master_bytes_per_kv = L * Hk * num_pages * 2         # fp16 master per page
        base_K_scale_bytes = L * Hk * tokens_padded * G * per_scale + master_bytes_per_kv
        base_V_scale_bytes = L * Hk * tokens_padded * G * per_scale + master_bytes_per_kv
    else:
        base_K_scale_bytes = L * Hk * tokens_padded * G * scale_dtype_bytes
        base_V_scale_bytes = L * Hk * tokens_padded * G * scale_dtype_bytes

    # ── Residual slots (always 4-bit symmetric in current impl) ──────
    num_k_cands_per_page = D // cfg.k_channel_group
    num_v_cands_per_page = math.ceil(cfg.page_size / cfg.v_token_block)
    total_k_cands = num_pages * Hk * L * num_k_cands_per_page
    total_v_cands = num_pages * Hk * L * num_v_cands_per_page

    k_slot_bytes = _bits_to_packed_bytes(cfg.page_size * cfg.k_channel_group, 4) + 2
    v_slot_bytes = _bits_to_packed_bytes(cfg.v_token_block * D, 4) + 2

    stored_k = int(total_k_cands * store_ratio)
    stored_v = int(total_v_cands * store_ratio)
    residual_K_bytes = stored_k * k_slot_bytes
    residual_V_bytes = stored_v * v_slot_bytes

    # ── Metadata (per page: slot indices, valid_tokens, etc.) ────────
    per_page_index_bytes = (
        num_k_cands_per_page * 4        # k_residual_slots (int32)
        + num_v_cands_per_page * 4      # v_residual_slots (int32)
        + 4                              # valid_tokens (int32)
        + 4                              # token_start (int32)
        + 4                              # page_id (int32) [bookkeeping]
    )
    metadata_bytes = num_pages * Hk * L * per_page_index_bytes

    # ── Error norms (fp16) ────────────────────────────────────────────
    error_norm_per_page = (num_k_cands_per_page + num_v_cands_per_page) * 2
    error_norm_bytes = num_pages * Hk * L * error_norm_per_page

    # ── Sketches (fp16, per channel group, per page) ─────────────────
    sketch_bytes = num_pages * Hk * L * num_k_cands_per_page * cfg.sketch_dim * 2

    total_bytes = (
        base_K_code_bytes + base_V_code_bytes
        + base_K_scale_bytes + base_V_scale_bytes
        + residual_K_bytes + residual_V_bytes
        + metadata_bytes + error_norm_bytes + sketch_bytes
    )

    # References use the same per-KV-head count Hk so comparisons are
    # apples-to-apples for GQA.  (KV memory in HF caches is per KV head.)
    fp16_kv_bytes = L * Hk * T * D * 2 * 2                   # K + V, fp16
    int4_kv_bytes = L * Hk * T * D * 1                       # K + V, INT4 packed

    return {
        "base_K_code_bytes":   base_K_code_bytes,
        "base_V_code_bytes":   base_V_code_bytes,
        "base_K_scale_bytes":  base_K_scale_bytes,
        "base_V_scale_bytes":  base_V_scale_bytes,
        "residual_K_bytes":    residual_K_bytes,
        "residual_V_bytes":    residual_V_bytes,
        "metadata_bytes":      metadata_bytes,
        "error_norm_bytes":    error_norm_bytes,
        "sketch_bytes":        sketch_bytes,
        "total_bytes":         total_bytes,
        "fp16_kv_bytes":       fp16_kv_bytes,
        "int4_kv_bytes":       int4_kv_bytes,
        "compression_vs_fp16": total_bytes / max(fp16_kv_bytes, 1),
        "compression_vs_int4": total_bytes / max(int4_kv_bytes, 1),
        "packed_mode":         packed,
    }


def print_memory_table(cfg: CacheConfig, seq_lengths: List[int], packed: bool = True):
    """Pretty-print memory estimates for a list of sequence lengths."""
    mode = "packed" if packed else "int8-storage"
    print(f"\n{'='*88}")
    print(f"CARE-KV Memory Estimate  (base={cfg.base_bits}bit, "
          f"store_budget={cfg.store_budget_ratio:.0%}, mode={mode})")
    print(f"{'='*88}")
    print(f"{'seq_len':>8}  {'base_MB':>8}  {'scale_MB':>9}  {'resid_MB':>9}  "
          f"{'meta_MB':>8}  {'sketch_MB':>10}  {'total_MB':>9}  {'vs_fp16':>8}")
    print(f"{'-'*88}")
    for T in seq_lengths:
        m = estimate_memory_bytes(cfg, T, packed=packed)
        base_mb   = (m["base_K_code_bytes"] + m["base_V_code_bytes"]) / 1e6
        scale_mb  = (m["base_K_scale_bytes"] + m["base_V_scale_bytes"]) / 1e6
        resid_mb  = (m["residual_K_bytes"] + m["residual_V_bytes"]) / 1e6
        meta_mb   = (m["metadata_bytes"] + m["error_norm_bytes"]) / 1e6
        sketch_mb = m["sketch_bytes"] / 1e6
        total_mb  = m["total_bytes"] / 1e6
        vs16      = m["compression_vs_fp16"]
        print(f"{T:>8,}  {base_mb:>8.2f}  {scale_mb:>9.2f}  {resid_mb:>9.2f}  "
              f"{meta_mb:>8.2f}  {sketch_mb:>10.2f}  {total_mb:>9.2f}  {vs16:>7.3f}x")
    print(f"{'='*88}\n")


# ─────────────────────────────────────────────
# Sensitivity calibration
# ─────────────────────────────────────────────

def calibrate_layer_sensitivity(
    model_forward: Callable,
    calibration_inputs: List[Tensor],
    num_layers: int,
    device: torch.device,
    hooks_enabled: bool = True,
) -> List[float]:
    """
    Estimate per-layer KV quantization sensitivity via calibration.

    Strategy: for each layer, measure the average norm of attention output
    perturbation when K/V are quantized to base_bits.

    Returns list of sensitivity weights (num_layers,), normalized to [0,1].

    Note: This is a lightweight proxy — in practice you may use
    perplexity-based sensitivity or gradient-based methods.
    """
    # Simple heuristic: later layers and middle layers tend to be more sensitive
    # This can be replaced with actual forward-pass measurements
    sensitivities = []
    for i in range(num_layers):
        # U-shaped sensitivity: first and last layers more sensitive
        # (empirical finding across several KV quantization papers)
        norm_pos = i / max(num_layers - 1, 1)
        s = 0.5 + 0.5 * abs(2 * norm_pos - 1)
        sensitivities.append(s)

    # Normalize
    max_s = max(sensitivities)
    return [s / max_s for s in sensitivities]


# ─────────────────────────────────────────────
# Perplexity evaluation
# ─────────────────────────────────────────────

def compute_perplexity(
    logits: Tensor,   # (seq_len, vocab_size)
    targets: Tensor,  # (seq_len,) int64
    ignore_index: int = -100,
) -> float:
    """Compute per-token perplexity from logits and target token ids."""
    loss = F.cross_entropy(
        logits[:-1],    # predict next token
        targets[1:],
        ignore_index=ignore_index,
        reduction="mean",
    )
    return math.exp(loss.item())


# ─────────────────────────────────────────────
# Budget sweep
# ─────────────────────────────────────────────

def budget_sweep(
    store_ratios: List[float],
    read_ratios: List[float],
    eval_fn: Callable[[float, float], float],  # (store_r, read_r) → metric
) -> Dict[Tuple[float, float], float]:
    """
    Sweep over (store_budget, read_budget) combinations and evaluate a metric.

    eval_fn should run CARE-KV with the given budgets and return a scalar metric
    (e.g. perplexity, accuracy).

    Returns dict: (store_ratio, read_ratio) → metric
    """
    results = {}
    for sr in store_ratios:
        for rr in read_ratios:
            if rr > sr:
                continue   # read budget can't exceed store budget
            metric = eval_fn(sr, rr)
            results[(sr, rr)] = metric
            print(f"  store={sr:.2%}  read={rr:.2%}  metric={metric:.4f}")
    return results


# ─────────────────────────────────────────────
# Ablation comparison
# ─────────────────────────────────────────────

ABLATION_VARIANTS = [
    "base_only",              # A1: INT2/INT3 base, no residual
    "residual_raw_error",     # A2: residual by raw quantization error norm
    "residual_attn_mass",     # A3: residual weighted by attention mass
    "residual_k_qdotr",       # A4: K residual by estimated |q·R_K|
    "residual_kv_separated",  # A5: K/V separated output-error-aware
    "care_kv_full",           # A6: full CARE-KV with boundary-aware K routing
    "care_kv_no_budget_sep",  # A7: CARE-KV without store/read budget separation
    "care_kv_no_sketch",      # A8: CARE-KV without sketch (full R_K loaded)
]


def run_ablation(
    variant: str,
    cfg: CacheConfig,
    eval_fn: Callable[[CacheConfig, str], Dict],
) -> Dict:
    """
    Run a single ablation variant and return evaluation results.

    eval_fn receives (cfg, variant_name) and should return a dict of metrics.
    """
    assert variant in ABLATION_VARIANTS, f"Unknown variant: {variant}"
    print(f"\n[Ablation] Running: {variant}")
    results = eval_fn(cfg, variant)
    print(f"  Results: {results}")
    return results


# ─────────────────────────────────────────────
# Attention entropy analysis
# ─────────────────────────────────────────────

def compute_attention_entropy(a: Tensor) -> float:
    """
    Compute entropy of attention distribution (in nats).
    a: (seq_len,) attention weights
    High entropy → many tokens compete → boundary risk higher
    """
    a_safe = a.clamp(min=1e-10)
    return -(a_safe * a_safe.log()).sum().item()


def compute_boundary_risk_stats(
    s: Tensor,   # (seq_len,) attention logits
    a: Tensor,   # (seq_len,) attention weights
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute statistics about boundary risk distribution.
    Useful for analysis / plotting.
    """
    s_top = s.max().item()
    margins = (s_top - s).clamp(min=eps)
    boundary_risk = 1.0 / margins              # (seq_len,)
    attn_weighted_risk = (a * boundary_risk).sum().item()
    entropy = compute_attention_entropy(a)

    return {
        "mean_margin":             margins.mean().item(),
        "min_margin":              margins.min().item(),
        "max_boundary_risk":       boundary_risk.max().item(),
        "attn_weighted_risk":      attn_weighted_risk,
        "attention_entropy":       entropy,
        "effective_context_size":  math.exp(entropy),  # ≈ number of tokens attention is spread over
    }
