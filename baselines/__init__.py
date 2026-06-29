"""baselines/ — same-condition SOTA KV-cache method reimplementations.

Every adapter in this package runs through the SAME unified
`KVMethodAdapter` interface from `common.py`, so the only thing that
varies across cells in the direct comparison is the per-method storage
treatment of K and V. Model loading, tokenization, dataset windowing,
PPL computation, and memory estimation are all shared.

Adapter status (as of `feat/sota-direct-comparison`):
  FP16Adapter        — works (reference; no K/V modification)
  BaseQuantAdapter   — works (uses existing CARE-KV base_quant path)
  CAREKVAdapter      — works (uses existing CARE-KV paper-best path)
  KIVIStyleAdapter   — works (same-condition reimplementation; per-channel K, per-token V)
  KVQuantStyleAdapter — STUB (blocker: pre-RoPE K storage; see sota_official_integration_status.md)
  MiKVStyleAdapter   — STUB (blocker: per-token bit-width plumbing)
  ZipCacheStyleAdapter — STUB (blocker: per-token bit-width + saliency pass)
"""

from .common import (
    KVMethodAdapter, ResultRow, eval_ppl_wikitext, eval_ppl_synthetic,
    fp16_kv_mb,
)
from .fp16_adapter import FP16Adapter
from .basequant_adapter import BaseQuantAdapter
from .carekv_adapter import CAREKVAdapter
from .kivi_style_adapter import KIVIStyleAdapter
from .kvquant_style_adapter import KVQuantStyleAdapter
from .mikv_style_adapter import MiKVStyleAdapter
from .zipcache_style_adapter import ZipCacheStyleAdapter
# Phase Q: base-quantizer interface + standalone quantizer instances
from .quantizer_base import BaseKVQuantizer, BaseKVQuantResult
from .uniform_quantizer import UniformBaseQuantizer
from .kivi_style_quantizer import KIVIStyleQuantizer

# Note: base-quantizer-expansion adapters (KVQuant-style /
# RotateKV-style same-condition reimplementations) live in
# baselines/kvquant_style.py and baselines/rotatekv_style.py. They are
# imported directly by tools/eval_base_quantizer_expansion.py rather
# than re-exported here, to avoid colliding with the existing
# `KVQuantStyleAdapter` stub from baselines/kvquant_style_adapter.py.
# The new working adapters share the kivi_style side-buffer dispatch
# in CARE-KV's cache; only the K/V quant function differs.

__all__ = [
    "KVMethodAdapter", "ResultRow", "eval_ppl_wikitext", "eval_ppl_synthetic",
    "fp16_kv_mb",
    "FP16Adapter", "BaseQuantAdapter", "CAREKVAdapter",
    "KIVIStyleAdapter", "KVQuantStyleAdapter",
    "MiKVStyleAdapter", "ZipCacheStyleAdapter",
    "BaseKVQuantizer", "BaseKVQuantResult",
    "UniformBaseQuantizer", "KIVIStyleQuantizer",
]
