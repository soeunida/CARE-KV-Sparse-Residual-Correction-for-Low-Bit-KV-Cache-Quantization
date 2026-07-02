"""
CARE-KV: Attention-Output-Error-Aware Sparse Residual KV Cache Correction
==========================================================================

Core pipeline:
  CacheConfig        → configuration
  CAREKVCache        → buffer manager
  ResidualStoreManager → store-time residual filtering
  ResidualRouter     → decode-time residual routing
  CAREKVAttention    → single-head attention with correction
  CAREKVLayer        → full prefill + decode layer
  patch_llama_model  → HuggingFace LLaMA integration
"""

from .cache import (
    CacheConfig, CAREKVCache, PageMeta,
    apply_carekv_env_overrides, layer_budget_multiplier,
)
from .quantizer import (
    QuantConfig, KQuantizer, VQuantizer,
    quantize, dequantize, quantize_and_residual,
    pack_int2, unpack_int2, pack_int3, unpack_int3, pack_int4, unpack_int4,
    pack_int2_2d, unpack_int2_2d, pack_int3_2d, unpack_int3_2d,
    pack_int4_2d, unpack_int4_2d,
    pack_codes_2d, unpack_codes_2d, packed_row_bytes,
)
from .residual_store import ResidualStoreManager, pack_4bit, unpack_4bit
from .residual_router import ResidualRouter
from .attention import CAREKVAttention, CAREKVMultiHeadAttention, apply_slot_corrections
from .layer import CAREKVLayer, get_debug_stats, reset_debug_stats, get_imp_per_layer
from .llama_patch import CAREKVLlamaAttention, patch_llama_model, reset_all_caches
from .utils import (
    estimate_memory_bytes,
    print_memory_table,
    calibrate_layer_sensitivity,
    compute_perplexity,
    budget_sweep,
    compute_boundary_risk_stats,
    ABLATION_VARIANTS,
)

__version__ = "0.2.0"
__all__ = [
    "CacheConfig", "CAREKVCache", "PageMeta",
    "apply_carekv_env_overrides", "layer_budget_multiplier",
    "QuantConfig", "KQuantizer", "VQuantizer",
    "quantize", "dequantize", "quantize_and_residual",
    "pack_int2", "unpack_int2", "pack_int3", "unpack_int3",
    "pack_int4", "unpack_int4",
    "pack_int2_2d", "unpack_int2_2d", "pack_int3_2d", "unpack_int3_2d",
    "pack_int4_2d", "unpack_int4_2d",
    "pack_codes_2d", "unpack_codes_2d", "packed_row_bytes",
    "ResidualStoreManager", "pack_4bit", "unpack_4bit",
    "ResidualRouter",
    "CAREKVAttention", "CAREKVMultiHeadAttention", "apply_slot_corrections",
    "CAREKVLayer", "get_debug_stats", "reset_debug_stats",
    "CAREKVLlamaAttention", "patch_llama_model", "reset_all_caches",
    "estimate_memory_bytes", "print_memory_table",
    "calibrate_layer_sensitivity", "compute_perplexity",
    "budget_sweep", "compute_boundary_risk_stats",
    "ABLATION_VARIANTS",
]
