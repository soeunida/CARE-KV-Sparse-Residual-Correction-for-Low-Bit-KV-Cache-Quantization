"""
care_kv/cache.py
----------------
KV Cache manager for CARE-KV.

Layout (v2: KV-head indexed, with valid_tokens tracking)
--------------------------------------------------------
Base KV (per KV head, not per query head — GQA-aware):
    base_K_codes[L, Hkv, P, T, D]  int8
    base_K_scale[L, Hkv, P, T, G]  fp16
    base_V_codes[L, Hkv, P, T, D]  int8
    base_V_scale[L, Hkv, P, T, G]  fp16

valid_tokens[L, Hkv, P]            int32
    Number of real (non-padded) tokens in each page.  read_base_concat()
    uses this to return only-valid-token tensors.

Residual KV:
    K residual slot:  (page_size, k_channel_group) → 4-bit packed
    V residual slot:  (v_token_block, head_dim)     → 4-bit packed

PageMetaTable:
    Per (layer, kv_head, page_id) metadata for routing.

Note on RoPE:
    The caller is responsible for storing post-RoPE K (so that base attention
    against K_hat is dimensionally correct).  V is stored unrotated.
"""

from __future__ import annotations
import os
import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math

from .quantizer import pack_codes_2d, unpack_codes_2d, packed_row_bytes


# ─────────────────────────────────────────────
# Env-variable overrides for CacheConfig kwargs
#
# Lets scripts honor CAREKV_PAGE_SIZE / CAREKV_V_TOKEN_BLOCK /
# CAREKV_K_CHANNEL_GROUP / CAREKV_SKETCH_DIM / CAREKV_GROUP_SIZE /
# CAREKV_STORE_BUDGET / CAREKV_READ_BUDGET / CAREKV_PACKED_BASE /
# CAREKV_SCALE_DTYPE / CAREKV_SCALE_QUANT without rebuilding configs by hand.
# Call this on the **kwargs dict** before constructing CacheConfig.
# ─────────────────────────────────────────────

def apply_carekv_env_overrides(kwargs: dict) -> dict:
    """Mutate `kwargs` in place to honor CAREKV_* env vars, then return it."""
    def _ovr_int(env, key):
        if env in os.environ and os.environ[env] != "":
            kwargs[key] = int(os.environ[env])
    def _ovr_float(env, key):
        if env in os.environ and os.environ[env] != "":
            kwargs[key] = float(os.environ[env])
    def _ovr_bool(env, key):
        if env in os.environ and os.environ[env] != "":
            kwargs[key] = os.environ[env] == "1"
    def _ovr_str(env, key):
        if env in os.environ and os.environ[env] != "":
            kwargs[key] = os.environ[env]

    _ovr_int("CAREKV_PAGE_SIZE",        "page_size")
    _ovr_int("CAREKV_V_TOKEN_BLOCK",    "v_token_block")
    _ovr_int("CAREKV_K_CHANNEL_GROUP",  "k_channel_group")
    _ovr_int("CAREKV_SKETCH_DIM",       "sketch_dim")
    _ovr_int("CAREKV_GROUP_SIZE",       "group_size")
    _ovr_int("CAREKV_MAX_PAGES",        "max_pages")
    _ovr_int("CAREKV_BASE_BITS",        "base_bits")
    _ovr_int("CAREKV_RESIDUAL_BITS",    "residual_bits")
    _ovr_float("CAREKV_STORE_BUDGET",   "store_budget_ratio")
    _ovr_float("CAREKV_READ_BUDGET",    "read_budget_ratio")
    _ovr_bool("CAREKV_PACKED_BASE",     "packed_base")
    _ovr_str("CAREKV_SCALE_DTYPE",      "scale_dtype")
    _ovr_str("CAREKV_SCALE_QUANT",      "scale_quant")

    # P1 + P2 + P4 + P5 knobs
    _ovr_str("CAREKV_ROUTE_POLICY",     "route_policy")
    _ovr_str("CAREKV_STORE_BUDGET_MODE","store_budget_mode")
    _ovr_str("CAREKV_READ_BUDGET_MODE", "read_budget_mode")
    _ovr_float("CAREKV_STORE_BUDGET_K", "store_budget_ratio_k")
    _ovr_float("CAREKV_STORE_BUDGET_V", "store_budget_ratio_v")
    _ovr_float("CAREKV_READ_BUDGET_K",  "read_budget_ratio_k")
    _ovr_float("CAREKV_READ_BUDGET_V",  "read_budget_ratio_v")
    _ovr_int("CAREKV_STORE_ABS_K",      "store_abs_k")
    _ovr_int("CAREKV_STORE_ABS_V",      "store_abs_v")
    _ovr_int("CAREKV_READ_ABS_K",       "read_abs_k")
    _ovr_int("CAREKV_READ_ABS_V",       "read_abs_v")
    _ovr_str("CAREKV_CORRECTION_IMPL",  "correction_impl")
    _ovr_str("CAREKV_PACK_IMPL",        "pack_impl")
    _ovr_str("CAREKV_BUDGET_POLICY",    "budget_policy")
    _ovr_bool("CAREKV_VDOM_OPTIMIZED",  "vdom_optimized")
    # Adaptive read-budget controls (default = no-op; preserves paper-best)
    _ovr_float("CAREKV_READ_RELATIVE_THRESHOLD", "read_relative_threshold")
    _ovr_float("CAREKV_READ_ABSOLUTE_THRESHOLD", "read_absolute_threshold")
    _ovr_int(  "CAREKV_READ_MIN_KEEP",           "read_min_keep")
    _ovr_float("CAREKV_READ_SCORE_TEMPERATURE",  "read_score_temperature")
    _ovr_float("CAREKV_CORRECTION_NORM_CLIP",    "correction_norm_clip")
    # Phase Q: base-quantizer selection (default "uniform" = paper-best,
    # bit-for-bit). KIVI-style integration wired through layer.py +
    # cache.py — see summaries/carekv_on_base_quantizers.md.
    _ovr_str(  "CAREKV_BASE_QUANTIZER",          "base_quantizer")
    _ovr_int(  "CAREKV_K_BITS",                  "k_bits_override")
    _ovr_int(  "CAREKV_V_BITS",                  "v_bits_override")
    # KVQuant-unblock: K store-mode for the kvquant_style base quantizer.
    #   "post_rope" (default) — quantize the post-RoPE K (KIVI-equivalent alias).
    #   "pre_rope"            — quantize K BEFORE RoPE (true KVQuant trait), then
    #                           re-apply RoPE so the residual lands post-RoPE.
    _ovr_str(  "CAREKV_K_STORE_MODE",            "k_store_mode")
    # Phase M: routing-baseline ablation knob (default "carekv" = paper-best score)
    _ovr_str(  "CAREKV_BASELINE_SCORE",          "baseline_score")
    # Comma-separated per-layer sensitivity, e.g. "1.5,1.2,1.0,0.8,..."
    if "CAREKV_LAYER_SENSITIVITY" in os.environ and os.environ["CAREKV_LAYER_SENSITIVITY"]:
        try:
            kwargs["layer_sensitivity"] = [
                float(x) for x in os.environ["CAREKV_LAYER_SENSITIVITY"].split(",")
                if x.strip()
            ]
        except ValueError:
            pass
    apply_determinism_env()
    return kwargs


_DETERMINISM_APPLIED = False


def apply_determinism_env():
    """Operationalize the proven chunked-correction exactness config: when
    CAREKV_DETERMINISTIC=1 or CAREKV_DISABLE_TF32=1, turn TF32 OFF and request
    deterministic algorithms. The chunked-correction equivalence debug showed
    TF32 (batch-size-dependent rounding) was the main source of chunked-vs-full
    divergence; with TF32 off + deterministic, chunk_size≥aligned is bit-exact.
    Idempotent; safe to call on every patch."""
    global _DETERMINISM_APPLIED
    if _DETERMINISM_APPLIED:
        return
    if os.environ.get("CAREKV_DETERMINISTIC") == "1" or os.environ.get("CAREKV_DISABLE_TF32") == "1":
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if os.environ.get("CAREKV_DETERMINISTIC") == "1":
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except Exception:
                    pass
            _DETERMINISM_APPLIED = True
        except Exception:
            pass


# ─────────────────────────────────────────────
# Per-layer budget multiplier (Phase E)
#
# Returns a float multiplier; multiplying the global per-kind budget by this
# value and rounding gives the per-layer budget.  The multiplier function is
# normalized over all layers so the *average* multiplier is 1.0, i.e. the
# total budget across layers is preserved.
# ─────────────────────────────────────────────

def _u_shape_multipliers(L: int):
    """Edge-heavy U-shape, mean-normalized to 1.0."""
    if L <= 1:
        return [1.0]
    raw = []
    for l in range(L):
        norm_pos = l / (L - 1)              # in [0, 1]
        # peaks at edges (norm_pos ∈ {0, 1}) → value 2.0; mid (0.5) → 0.5
        w = 0.5 + 1.5 * abs(2 * norm_pos - 1)
        raw.append(w)
    mean_w = sum(raw) / L
    return [w / mean_w for w in raw]


def layer_budget_multiplier(cfg: "CacheConfig", layer_id: int) -> float:
    """Return the per-layer budget scaling factor for `layer_id` under
    cfg.budget_policy.  Always positive; mean across layers ≈ 1.0."""
    policy = getattr(cfg, "budget_policy", "uniform")
    L = cfg.num_layers
    if policy == "uniform" or L <= 1:
        return 1.0
    if policy == "u_shaped":
        return _u_shape_multipliers(L)[layer_id]
    if policy == "sensitivity":
        s = cfg.layer_sensitivity or [1.0] * L
        mean_s = sum(s) / max(L, 1)
        if mean_s <= 0:
            return 1.0
        return s[layer_id] / mean_s
    if policy == "layer_head_impact":
        # NOT YET LIVE: true output-impact allocation needs a calibration pass
        # (measure per-layer/head impact, then allocate_impact_budget(impact,
        # total)). The allocate_impact_budget()/attention_output_residual_scores()
        # helpers are unit-tested but not wired into a two-pass run, so this
        # falls back to uniform. Honestly reported as uniform-equivalent.
        return 1.0
    return 1.0


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class CacheConfig:
    num_layers: int = 32
    num_heads: int = 32                    # number of query heads
    num_kv_heads: Optional[int] = None     # number of KV heads (GQA); None → num_heads
    head_dim: int = 128
    page_size: int = 16          # tokens per page
    max_pages: int = 512         # max pages per (layer, kv_head)

    # Quantization
    base_bits: int = 2           # INT2 / INT3 / INT4 for base KV
    residual_bits: int = 4       # INT4 for residual
    group_size: int = 64         # quantization group size

    # Residual granularity
    k_channel_group: int = 32    # K residual covers this many channels
    v_token_block: int = 4       # V residual covers this many consecutive tokens

    # Budget
    store_budget_ratio: float = 0.10   # store up to this fraction of candidates
    read_budget_ratio: float = 0.03    # read up to this fraction per decode step

    # Sketch
    sketch_dim: int = 32         # R_K sketch dim; =k_channel_group → full-rank
                                 # (no lossy projection) so |q·R_K| ranking is
                                 # exact. 16 was lossy (32→16); bump validated
                                 # (router_diagnostic: sk16 14.98 → sk32 14.93 →
                                 # sk64 14.81 at N=8 SL=128; 32 = full rank).

    # Sensitivity (per-layer weights, can be calibrated)
    layer_sensitivity: Optional[List[float]] = None

    # Storage mode: when False, base K/V codes are stored as int8 (fast
    # accessor, oversized memory).  When True, they are packed to
    # `base_bits`-wide tight bytes — actual on-device memory matches the
    # estimator's packed mode at the cost of pack/unpack on every page op.
    packed_base: bool = False
    # Retained for backwards compatibility; same meaning as packed_base.
    packed_storage: bool = False

    # Scale storage:
    #   scale_dtype  : "fp16" (default) / "bf16" / "fp32" — width of base scale buffers
    #   scale_quant  : "none" (default) / "int8" — experimental.  When "int8",
    #                  per-(layer, KV head, page) scale tensors are quantized
    #                  to int8 with one fp16 master scale per page.  Cuts
    #                  scale memory ~50% on TinyLlama configs but is OFF by
    #                  default — verify PPL impact before enabling.
    scale_dtype: str = "fp16"
    scale_quant: str = "none"

    # Routing / budget knobs (P1 + P2).  All optional; when None or 0 the
    # legacy `store_budget_ratio` / `read_budget_ratio` are used instead.
    #
    #   store/read_budget_mode      : "ratio" (default) or "absolute"
    #   store/read_budget_ratio_k/v : per-kind ratio override
    #   store/read_abs_k/v          : per-kind absolute slot count override
    #   route_policy                : "joint" / "separate" (default) / "k_first" / "adaptive"
    store_budget_mode: str = "ratio"
    read_budget_mode: str = "ratio"
    store_budget_ratio_k: Optional[float] = None
    store_budget_ratio_v: Optional[float] = None
    read_budget_ratio_k: Optional[float] = None
    read_budget_ratio_v: Optional[float] = None
    store_abs_k: int = 0
    store_abs_v: int = 0
    read_abs_k: int = 0
    read_abs_v: int = 0
    route_policy: str = "separate"

    # Optimized Vdom (V-only) deployment. Default False = paper-best behavior
    # (K residual arena always pre-allocated). When True, the cache does NOT
    # pre-allocate a K residual arena (a 1-slot stub is kept so indexing code
    # stays valid) and alloc_k_slot() raises — proving, as an audit guard, that
    # a Vdom deployment stores zero K residuals. Only safe with a V-only store
    # path (CAREKV_PREFILL_RESIDUAL_KIND=v, STORE_ABS_K=0, READ_ABS_K=0).
    vdom_optimized: bool = False

    # Phase N+: adaptive read-budget controls (default = off; preserves
    # paper-best behavior). When read_budget_mode == "adaptive_score",
    # read_abs_k/v become MAX caps and the router additionally filters
    # selected slots by score threshold (relative + absolute) before
    # applying corrections. read_min_keep is a floor on slot count when
    # any candidate exists. read_score_temperature is reserved (currently
    # unused) for future softmax-style probabilistic gating.
    #
    #   read_budget_mode in {"ratio", "absolute", "adaptive_score"}
    read_relative_threshold: float = 0.0
    read_absolute_threshold: float = 0.0
    read_min_keep: int = 0
    read_score_temperature: float = 1.0

    # Phase N+: optional post-correction safety guard. When > 0, the
    # per-(layer, kv_head) cached/python correction routine will clip the
    # L2 norm of (ΔO_K, ΔO_V) to this threshold and increment a debug
    # counter. Default 0.0 (disabled).
    correction_norm_clip: float = 0.0

    # Phase Q: base-quantizer selection (default "uniform" = paper-best
    # path, bit-for-bit). KIVI-style integration wired through layer.py +
    # cache.py — see summaries/carekv_on_base_quantizers.md.
    #
    #   base_quantizer in {"uniform", "kivi_style"}
    base_quantizer: str = "uniform"
    # Optional per-kind bit-width overrides; if -1 fall back to base_bits.
    k_bits_override: int = -1
    v_bits_override: int = -1
    # KVQuant-unblock: K store-mode (only meaningful for base_quantizer ==
    # "kvquant_style"). "post_rope" (default) keeps the prior behavior
    # bit-for-bit (the kvquant_style dispatch aliases the post-RoPE KIVI
    # path). "pre_rope" quantizes K in pre-RoPE coordinates (the true
    # KVQuant trait) and re-applies RoPE to K_hat so the CARE-KV residual
    # is still computed in the post-RoPE coordinate system the correction
    # path reads. See layer.py:prefill / decode_step.
    k_store_mode: str = "post_rope"

    # Phase M routing-baseline ablation hook: which scoring formula the
    # ResidualRouter uses to rank K / V candidates. "carekv" preserves the
    # paper-best score exactly; the other values exist for ablation only.
    #   "carekv"          (default — paper-best, do not change)
    #   "random"          uniform random scores (no scoring signal)
    #   "magnitude_only"  only the residual-magnitude factor (K: |q·R_K|; V: ||R_V||)
    #   "attention_only"  only the attention-mass factor
    #   "oracle_proxy"    magnitude × attention (drops the structural-prior +
    #                     sensitivity multipliers); diagnostic upper bound
    baseline_score: str = "carekv"

    # Runtime / implementation switches (P4, P5).
    correction_impl: str = "cached"          # "python" | "cached"
    pack_impl: str = "vectorized"            # "python" | "vectorized"

    # Phase E: per-layer budget allocation policy.
    #
    #   uniform    : every layer gets the global per-kind budget (default).
    #   u_shaped   : multiplier = 0.5 + 1.5·|2·l/(L−1) − 1|, normalized so the
    #                mean across layers is 1.0 → total budget preserved.  Early
    #                and late layers get up to ~2× while middle layers get
    #                ~0.5× (typical KV-quantization sensitivity profile).
    #   sensitivity: multiplier = layer_sensitivity[l] / mean(layer_sensitivity).
    #                If layer_sensitivity is all 1.0 this degenerates to uniform;
    #                provide non-uniform values via the `layer_sensitivity` cfg
    #                kwarg or CAREKV_LAYER_SENSITIVITY env (comma-separated).
    budget_policy: str = "uniform"

    def __post_init__(self):
        if self.layer_sensitivity is None:
            self.layer_sensitivity = [1.0] * self.num_layers
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        # Honor either flag; packed_base is the canonical name.
        if self.packed_storage and not self.packed_base:
            self.packed_base = True
        if self.scale_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"scale_dtype must be fp16/bf16/fp32, got {self.scale_dtype}")
        if self.scale_quant not in {"none", "int8"}:
            raise ValueError(f"scale_quant must be none/int8, got {self.scale_quant}")
        if self.store_budget_mode not in {"ratio", "absolute"}:
            raise ValueError(f"store_budget_mode={self.store_budget_mode}")
        if self.read_budget_mode not in {"ratio", "absolute", "adaptive_score"}:
            raise ValueError(f"read_budget_mode={self.read_budget_mode}")
        if self.route_policy not in {"joint", "separate", "k_first", "adaptive"}:
            raise ValueError(f"route_policy={self.route_policy}")
        if self.correction_impl not in {"python", "cached", "vectorized"}:
            raise ValueError(f"correction_impl={self.correction_impl}")
        if self.pack_impl not in {"python", "vectorized"}:
            raise ValueError(f"pack_impl={self.pack_impl}")
        if self.budget_policy not in {"uniform", "u_shaped", "sensitivity", "layer_head_impact"}:
            raise ValueError(f"budget_policy={self.budget_policy}")

    def scale_torch_dtype(self):
        """Return the torch dtype corresponding to scale_dtype."""
        import torch as _torch
        return {
            "fp16": _torch.float16,
            "bf16": _torch.bfloat16,
            "fp32": _torch.float32,
        }[self.scale_dtype]


# ─────────────────────────────────────────────
# Page Metadata
# ─────────────────────────────────────────────

@dataclass
class PageMeta:
    page_id: int
    layer_id: int
    kv_head: int             # KV head this page belongs to
    token_start: int         # absolute token position of first token in page
    num_tokens: int          # actual valid tokens in this page (≤ page_size)

    # Residual slot indices (-1 = no residual stored)
    k_residual_slots: List[int] = field(default_factory=list)   # one per k_channel_group
    v_residual_slots: List[int] = field(default_factory=list)   # one per v_token_block

    # Summary statistics (stored at creation time)
    k_error_norm: Optional[Tensor] = None    # (num_k_groups,)
    v_error_norm: Optional[Tensor] = None    # (num_v_blocks,)
    k_sketch: Optional[Tensor] = None        # (num_k_groups, sketch_dim)

    # Prior score (computed at store time)
    prior_score: float = 0.0


# ─────────────────────────────────────────────
# KV Cache Manager
# ─────────────────────────────────────────────

class CAREKVCache:
    """
    Full KV cache manager for one sequence.

    Layout is indexed by KV head, not query head — GQA-aware.
    Each page has a separately-tracked valid_tokens count so that partial
    last pages do not leak padded zeros into attention.
    """

    def __init__(self, cfg: CacheConfig, device: torch.device = torch.device("cpu")):
        self.cfg = cfg
        self.device = device
        self._init_buffers()

    def _init_buffers(self):
        cfg = self.cfg
        L, Hkv, P, T, D = (
            cfg.num_layers, cfg.num_kv_heads,
            cfg.max_pages, cfg.page_size, cfg.head_dim,
        )
        G = D // cfg.group_size
        dev = self.device

        # ── Base KV code buffers (KV-head indexed) ───────────────────
        # packed_base=False: int8 per code, shape (L, Hkv, P, T, D).
        # packed_base=True : packed, shape (L, Hkv, P, T, packed_row_bytes(D, base_bits)).
        # In both cases scales are fp16 (L, Hkv, P, T, G).
        if cfg.packed_base:
            self._packed_row_bytes = packed_row_bytes(D, cfg.base_bits)
            self.base_K_codes = torch.zeros(
                L, Hkv, P, T, self._packed_row_bytes, dtype=torch.int8, device=dev,
            )
            self.base_V_codes = torch.zeros(
                L, Hkv, P, T, self._packed_row_bytes, dtype=torch.int8, device=dev,
            )
        else:
            self._packed_row_bytes = None
            self.base_K_codes = torch.zeros(L, Hkv, P, T, D, dtype=torch.int8, device=dev)
            self.base_V_codes = torch.zeros(L, Hkv, P, T, D, dtype=torch.int8, device=dev)

        # Scale storage: either raw (chosen dtype) or int8-quantized per page
        # with one master scale per page.
        if cfg.scale_quant == "int8":
            # int8 codes share the (L, Hkv, P, T, G) layout but each byte
            # holds the quantized scale; one fp16 master per page covers all
            # T·G entries within that page.
            self.base_K_scale = torch.zeros(L, Hkv, P, T, G, dtype=torch.int8, device=dev)
            self.base_V_scale = torch.zeros(L, Hkv, P, T, G, dtype=torch.int8, device=dev)
            self.base_K_scale_master = torch.zeros(L, Hkv, P, dtype=torch.float16, device=dev)
            self.base_V_scale_master = torch.zeros(L, Hkv, P, dtype=torch.float16, device=dev)
        else:
            scale_dt = cfg.scale_torch_dtype()
            self.base_K_scale = torch.zeros(L, Hkv, P, T, G, dtype=scale_dt, device=dev)
            self.base_V_scale = torch.zeros(L, Hkv, P, T, G, dtype=scale_dt, device=dev)
            self.base_K_scale_master = None
            self.base_V_scale_master = None

        # ── Valid token counts per page ──────────────────────────────
        self.valid_tokens = torch.zeros(L, Hkv, P, dtype=torch.int32, device=dev)

        # ── Side-channel fp16 K_hat / V_hat buffer ───────────────────
        # Used by Phase Q-stacked (kivi_style) and the
        # base-quantizer-expansion (rotatekv_style, kvquant_style
        # post-RoPE variant). All three need a buffer that holds
        # already-dequantized K_hat / V_hat in fp16, because their
        # per-channel / per-token scale layouts don't fit the
        # (L,Hkv,P,T,G) layout used for uniform per-group quant.
        # Memory accounting reports the base scheme's theoretical bits;
        # the side-buffer's fp16 bytes are a prototype-implementation
        # cost (documented in
        # summaries/carekv_on_base_quantizers.md).
        _BASE_QUANTS_WITH_SIDE_BUFFER = {"kivi_style", "rotatekv_style",
                                         "randrot_style", "kvquant_style"}
        if getattr(cfg, "base_quantizer", "uniform") in _BASE_QUANTS_WITH_SIDE_BUFFER:
            self.base_K_hat_fp16 = torch.zeros(
                L, Hkv, P, T, D, dtype=torch.float16, device=dev,
            )
            self.base_V_hat_fp16 = torch.zeros(
                L, Hkv, P, T, D, dtype=torch.float16, device=dev,
            )
        else:
            self.base_K_hat_fp16 = None
            self.base_V_hat_fp16 = None

        # ── Residual slot buffers ────────────────────────────────────
        # Size the slot arena from the most-favorable of: ratio-budget or
        # absolute-budget worst case.  When `store_budget_mode == "absolute"`
        # the ratio is typically 0, so falling back to it would yield a
        # tiny arena (= 64) that fills up almost immediately under real
        # absolute budgets.
        ratio_estimate = int(P * Hkv * L * cfg.store_budget_ratio * 4)
        abs_estimate = int(P * Hkv * L * (cfg.store_abs_k + cfg.store_abs_v) * 2)
        max_slots = max(64, ratio_estimate, abs_estimate)
        k_slot_size = cfg.page_size * cfg.k_channel_group // 2   # 4-bit packed
        v_slot_size = cfg.v_token_block * D // 2
        # Optimized Vdom: do NOT pre-allocate a K residual arena. A V-only
        # (Vdom) deployment never writes or reads K residuals (store: use_k=0;
        # router: kind="v" forces bk=0 and skips K scoring), so the K arena is
        # dead memory. Keep a 1-slot stub so any index math stays valid, and
        # let alloc_k_slot() raise as an audit guard.
        self.vdom_optimized = bool(getattr(cfg, "vdom_optimized", False))
        k_slots = 1 if self.vdom_optimized else max_slots
        self.k_residual_buf = torch.zeros(k_slots, k_slot_size, dtype=torch.int8, device=dev)
        self.v_residual_buf = torch.zeros(max_slots, v_slot_size, dtype=torch.int8, device=dev)
        self.k_residual_scale = torch.zeros(k_slots, 1, dtype=torch.float16, device=dev)
        self.v_residual_scale = torch.zeros(max_slots, 1, dtype=torch.float16, device=dev)
        self._k_slot_free: List[int] = [] if self.vdom_optimized else list(range(max_slots))
        self._v_slot_free: List[int] = list(range(max_slots))

        # ── Page metadata table ──────────────────────────────────────
        self.meta_table: List[List[List[Optional[PageMeta]]]] = [
            [[None] * P for _ in range(Hkv)]
            for _ in range(L)
        ]

        # ── Page counters ────────────────────────────────────────────
        self.next_page: List[List[int]] = [[0] * Hkv for _ in range(L)]

    # ─────────────────────────────────────────
    # Allocation helpers
    # ─────────────────────────────────────────

    def alloc_page(self, layer: int, kv_head: int) -> int:
        pid = self.next_page[layer][kv_head]
        assert pid < self.cfg.max_pages, "KV cache is full"
        self.next_page[layer][kv_head] += 1
        return pid

    def alloc_k_slot(self) -> int:
        if getattr(self, "vdom_optimized", False):
            raise RuntimeError(
                "alloc_k_slot() called under vdom_optimized=True: a Vdom (V-only) "
                "deployment must not store K residuals. Set "
                "CAREKV_PREFILL_RESIDUAL_KIND=v and STORE_ABS_K=0, or disable "
                "CAREKV_VDOM_OPTIMIZED.")
        assert self._k_slot_free, "K residual buffer is full"
        return self._k_slot_free.pop(0)

    def alloc_v_slot(self) -> int:
        assert self._v_slot_free, "V residual buffer is full"
        return self._v_slot_free.pop(0)

    def free_k_slot(self, slot: int):
        self._k_slot_free.append(slot)

    def free_v_slot(self, slot: int):
        self._v_slot_free.append(slot)

    # ─────────────────────────────────────────
    # Streaming-decode helpers
    # ─────────────────────────────────────────

    def get_open_page(self, layer: int, kv_head: int) -> Optional[int]:
        """
        Return the id of the last allocated page if it still has room
        (valid_tokens < page_size), else None.
        """
        np = self.next_page[layer][kv_head]
        if np == 0:
            return None
        last = np - 1
        if int(self.valid_tokens[layer, kv_head, last].item()) < self.cfg.page_size:
            return last
        return None

    def page_valid(self, layer: int, kv_head: int, page_id: int) -> int:
        return int(self.valid_tokens[layer, kv_head, page_id].item())

    # ─────────────────────────────────────────
    # Base KV write
    # ─────────────────────────────────────────

    def _write_scale_page(self, dst_codes_view, dst_master, scale_in):
        """Helper: write a (T, G) scale tensor into the cache, honoring
        scale_dtype / scale_quant settings.  dst_codes_view points at the
        (T, G) slice of the scale buffer; dst_master at the (scalar) master
        cell when scale_quant == 'int8' (else ignored)."""
        if self.cfg.scale_quant == "int8":
            # Per-page master = max abs across (T·G).  qmax = 127 (signed int8).
            max_abs = scale_in.abs().max().clamp(min=1e-8).to(torch.float32)
            master = (max_abs / 127.0).to(torch.float16)
            codes = (scale_in.to(torch.float32) / master.to(torch.float32))
            codes = codes.round().clamp(-128, 127).to(torch.int8)
            dst_codes_view.copy_(codes)
            dst_master.copy_(master)
        else:
            dst_codes_view.copy_(scale_in.to(dst_codes_view.dtype))

    def _read_scale_page(self, src_codes_view, src_master):
        """Inverse of _write_scale_page; returns (T, G) tensor in fp32."""
        if self.cfg.scale_quant == "int8":
            codes = src_codes_view.to(torch.float32)
            master = src_master.to(torch.float32)
            return codes * master
        else:
            return src_codes_view.to(torch.float32)

    def write_base(
        self,
        layer: int,
        kv_head: int,
        page_id: int,
        K_codes: Tensor,   # (T, D) int8
        K_scale: Tensor,   # (T, G) scale_dtype
        V_codes: Tensor,   # (T, D) int8
        V_scale: Tensor,   # (T, G) scale_dtype
        num_valid: int,
    ):
        """
        Write a (zero-padded) page worth of base codes.  num_valid records
        how many of the leading T rows are real tokens.  When packed_base
        is enabled, K_codes / V_codes are packed before storage.
        Scales are written through _write_scale_page which honors
        scale_dtype / scale_quant.
        """
        assert 0 <= num_valid <= self.cfg.page_size, num_valid
        if self.cfg.packed_base:
            K_packed = pack_codes_2d(K_codes, self.cfg.base_bits)
            V_packed = pack_codes_2d(V_codes, self.cfg.base_bits)
            self.base_K_codes[layer, kv_head, page_id] = K_packed
            self.base_V_codes[layer, kv_head, page_id] = V_packed
        else:
            self.base_K_codes[layer, kv_head, page_id] = K_codes
            self.base_V_codes[layer, kv_head, page_id] = V_codes
        self._write_scale_page(
            self.base_K_scale[layer, kv_head, page_id],
            self.base_K_scale_master[layer, kv_head, page_id] if self.base_K_scale_master is not None else None,
            K_scale,
        )
        self._write_scale_page(
            self.base_V_scale[layer, kv_head, page_id],
            self.base_V_scale_master[layer, kv_head, page_id] if self.base_V_scale_master is not None else None,
            V_scale,
        )
        self.valid_tokens[layer, kv_head, page_id] = num_valid

    def append_to_page(
        self,
        layer: int,
        kv_head: int,
        page_id: int,
        K_codes_row: Tensor,    # (D,) int8 — single token
        K_scale_row: Tensor,    # (G,) scale_dtype
        V_codes_row: Tensor,    # (D,) int8
        V_scale_row: Tensor,    # (G,) scale_dtype
    ) -> int:
        """
        Append a single token's codes into an open page.  Returns the
        offset within the page where it was written.  For scale_quant=int8,
        the per-page master is held fixed at its current value (set by the
        first write_base) and incoming scales are clamped to fit; this
        matches typical decode usage where prefill already established the
        scale range for the page.
        """
        offset = int(self.valid_tokens[layer, kv_head, page_id].item())
        assert offset < self.cfg.page_size, "page already full"
        if self.cfg.packed_base:
            K_packed_row = pack_codes_2d(
                K_codes_row.unsqueeze(0), self.cfg.base_bits,
            ).squeeze(0)
            V_packed_row = pack_codes_2d(
                V_codes_row.unsqueeze(0), self.cfg.base_bits,
            ).squeeze(0)
            self.base_K_codes[layer, kv_head, page_id, offset] = K_packed_row
            self.base_V_codes[layer, kv_head, page_id, offset] = V_packed_row
        else:
            self.base_K_codes[layer, kv_head, page_id, offset] = K_codes_row
            self.base_V_codes[layer, kv_head, page_id, offset] = V_codes_row

        if self.cfg.scale_quant == "int8":
            k_master = self.base_K_scale_master[layer, kv_head, page_id].to(torch.float32).clamp(min=1e-8)
            v_master = self.base_V_scale_master[layer, kv_head, page_id].to(torch.float32).clamp(min=1e-8)
            k_codes = (K_scale_row.to(torch.float32) / k_master).round().clamp(-128, 127).to(torch.int8)
            v_codes = (V_scale_row.to(torch.float32) / v_master).round().clamp(-128, 127).to(torch.int8)
            self.base_K_scale[layer, kv_head, page_id, offset] = k_codes
            self.base_V_scale[layer, kv_head, page_id, offset] = v_codes
        else:
            scale_dt = self.base_K_scale.dtype
            self.base_K_scale[layer, kv_head, page_id, offset] = K_scale_row.to(scale_dt)
            self.base_V_scale[layer, kv_head, page_id, offset] = V_scale_row.to(scale_dt)
        self.valid_tokens[layer, kv_head, page_id] = offset + 1
        return offset

    # ─────────────────────────────────────────
    # Base KV read
    # ─────────────────────────────────────────

    def _maybe_unpack_rows(self, codes_rows: Tensor) -> Tensor:
        """Decode packed (n, packed_row_bytes) → (n, D) int8 if packed_base, else passthrough."""
        if self.cfg.packed_base and codes_rows.numel() > 0:
            return unpack_codes_2d(codes_rows, self.cfg.base_bits, self.cfg.head_dim)
        return codes_rows

    def _maybe_dequant_scale(self, scale_view, master_view, target_dtype):
        """Return scales in `target_dtype`; if scale_quant=int8, expand
        codes × master back to float."""
        if self.cfg.scale_quant == "int8":
            return (scale_view.to(torch.float32) * master_view.to(torch.float32)).to(target_dtype)
        return scale_view.to(target_dtype)

    def read_base_concat(
        self,
        layer: int,
        kv_head: int,
        page_ids: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Read base codes for the given page_ids, returning only the valid
        (non-padded) rows concatenated.  Also returns per-page valid lengths.

        With packed_base=True, codes are unpacked on the fly to (N_valid, D) int8.
        With scale_quant=int8, scales are expanded with their per-page master
        and returned in fp16.

        Returns
        -------
        K_codes : (N_valid, D) int8
        K_scale : (N_valid, G) fp16
        V_codes : (N_valid, D) int8
        V_scale : (N_valid, G) fp16
        valid_lens : list of per-page valid token counts (length = len(page_ids))
        """
        out_scale_dtype = torch.float16   # downstream code assumes fp16-compatible
        Kc_list, Ks_list, Vc_list, Vs_list = [], [], [], []
        valid_lens: List[int] = []
        for pid in page_ids:
            n = int(self.valid_tokens[layer, kv_head, pid].item())
            valid_lens.append(n)
            if n == 0:
                continue
            Kc_list.append(self._maybe_unpack_rows(self.base_K_codes[layer, kv_head, pid, :n]))
            Vc_list.append(self._maybe_unpack_rows(self.base_V_codes[layer, kv_head, pid, :n]))
            k_master = (self.base_K_scale_master[layer, kv_head, pid]
                        if self.cfg.scale_quant == "int8" else None)
            v_master = (self.base_V_scale_master[layer, kv_head, pid]
                        if self.cfg.scale_quant == "int8" else None)
            Ks_list.append(self._maybe_dequant_scale(
                self.base_K_scale[layer, kv_head, pid, :n], k_master, out_scale_dtype))
            Vs_list.append(self._maybe_dequant_scale(
                self.base_V_scale[layer, kv_head, pid, :n], v_master, out_scale_dtype))
        if not Kc_list:
            D = self.cfg.head_dim
            G = D // self.cfg.group_size
            empty_i8 = torch.zeros(0, D, dtype=torch.int8, device=self.device)
            empty_f = torch.zeros(0, G, dtype=torch.float16, device=self.device)
            return empty_i8, empty_f, empty_i8, empty_f, valid_lens
        return (
            torch.cat(Kc_list, dim=0),
            torch.cat(Ks_list, dim=0),
            torch.cat(Vc_list, dim=0),
            torch.cat(Vs_list, dim=0),
            valid_lens,
        )

    # Backward-compat wrapper: returns padded (per-page) tensors.
    # New code should prefer read_base_concat.
    def read_base(self, layer: int, kv_head: int, page_ids: List[int]):
        out_scale_dtype = torch.float16
        Kc_chunks, Vc_chunks, Ks_chunks, Vs_chunks = [], [], [], []
        for pid in page_ids:
            Kc_chunks.append(self._maybe_unpack_rows(self.base_K_codes[layer, kv_head, pid]))
            Vc_chunks.append(self._maybe_unpack_rows(self.base_V_codes[layer, kv_head, pid]))
            k_master = (self.base_K_scale_master[layer, kv_head, pid]
                        if self.cfg.scale_quant == "int8" else None)
            v_master = (self.base_V_scale_master[layer, kv_head, pid]
                        if self.cfg.scale_quant == "int8" else None)
            Ks_chunks.append(self._maybe_dequant_scale(
                self.base_K_scale[layer, kv_head, pid], k_master, out_scale_dtype))
            Vs_chunks.append(self._maybe_dequant_scale(
                self.base_V_scale[layer, kv_head, pid], v_master, out_scale_dtype))
        Kc = torch.cat(Kc_chunks, dim=0)
        Vc = torch.cat(Vc_chunks, dim=0)
        Ks = torch.cat(Ks_chunks, dim=0)
        Vs = torch.cat(Vs_chunks, dim=0)
        return Kc, Ks, Vc, Vs

    # ─────────────────────────────────────────
    # KIVI-style base K_hat / V_hat side-channel
    # ─────────────────────────────────────────

    def write_base_kivi(
        self,
        layer: int,
        kv_head: int,
        page_id: int,
        K_hat: Tensor,    # (T, D) fp16 — already KIVI-dequantized
        V_hat: Tensor,    # (T, D) fp16
        num_valid: int,
    ):
        """Stash dequantized K_hat / V_hat into the fp16 side buffer.
        Valid when cfg.base_quantizer uses the side-buffer dispatch
        (kivi_style / rotatekv_style / kvquant_style)."""
        assert self.base_K_hat_fp16 is not None, (
            "write_base_kivi requires cfg.base_quantizer in "
            "{kivi_style, rotatekv_style, randrot_style, kvquant_style}")
        assert 0 <= num_valid <= self.cfg.page_size, num_valid
        self.base_K_hat_fp16[layer, kv_head, page_id] = K_hat.to(torch.float16)
        self.base_V_hat_fp16[layer, kv_head, page_id] = V_hat.to(torch.float16)
        self.valid_tokens[layer, kv_head, page_id] = num_valid

    def append_to_page_kivi(
        self,
        layer: int,
        kv_head: int,
        page_id: int,
        K_hat_row: Tensor,    # (D,) fp16 — already KIVI-dequantized single token
        V_hat_row: Tensor,    # (D,) fp16
    ) -> int:
        """Decode-mode append: write one dequantized K/V row into the
        open page's side buffer."""
        assert self.base_K_hat_fp16 is not None, (
            "append_to_page_kivi requires cfg.base_quantizer in "
            "{kivi_style, rotatekv_style, randrot_style, kvquant_style}")
        offset = int(self.valid_tokens[layer, kv_head, page_id].item())
        assert offset < self.cfg.page_size, "page already full"
        self.base_K_hat_fp16[layer, kv_head, page_id, offset] = K_hat_row.to(torch.float16)
        self.base_V_hat_fp16[layer, kv_head, page_id, offset] = V_hat_row.to(torch.float16)
        self.valid_tokens[layer, kv_head, page_id] = offset + 1
        return offset

    def read_base_hat_concat(
        self,
        layer: int,
        kv_head: int,
        page_ids: List[int],
    ) -> Tuple[Tensor, Tensor, List[int]]:
        """Side-buffer counterpart of read_base_concat: returns valid
        K_hat / V_hat rows directly from the fp16 side buffer (no
        dequantization needed). Returns (K_hat, V_hat, valid_lens).
        Valid when cfg.base_quantizer uses the side-buffer dispatch."""
        assert self.base_K_hat_fp16 is not None, (
            "read_base_hat_concat requires cfg.base_quantizer in "
            "{kivi_style, rotatekv_style, randrot_style, kvquant_style}")
        Kh_list, Vh_list, valid_lens = [], [], []
        for pid in page_ids:
            n = int(self.valid_tokens[layer, kv_head, pid].item())
            valid_lens.append(n)
            if n == 0:
                continue
            Kh_list.append(self.base_K_hat_fp16[layer, kv_head, pid, :n])
            Vh_list.append(self.base_V_hat_fp16[layer, kv_head, pid, :n])
        if not Kh_list:
            D = self.cfg.head_dim
            empty = torch.zeros(0, D, dtype=torch.float16, device=self.device)
            return empty, empty, valid_lens
        return torch.cat(Kh_list, dim=0), torch.cat(Vh_list, dim=0), valid_lens

    # ─────────────────────────────────────────
    # Residual slot write / read
    # ─────────────────────────────────────────

    def write_k_residual(self, slot: int, packed: Tensor, scale: Tensor):
        self.k_residual_buf[slot] = packed.flatten()
        self.k_residual_scale[slot] = scale

    def write_v_residual(self, slot: int, packed: Tensor, scale: Tensor):
        self.v_residual_buf[slot] = packed.flatten()
        self.v_residual_scale[slot] = scale

    def read_k_residual(self, slot: int) -> Tuple[Tensor, Tensor]:
        return self.k_residual_buf[slot], self.k_residual_scale[slot]

    def read_v_residual(self, slot: int) -> Tuple[Tensor, Tensor]:
        return self.v_residual_buf[slot], self.v_residual_scale[slot]

    # ─────────────────────────────────────────
    # Page accounting
    # ─────────────────────────────────────────

    def num_pages(self, layer: int, kv_head: int) -> int:
        return self.next_page[layer][kv_head]

    def all_page_ids(self, layer: int, kv_head: int) -> List[int]:
        return list(range(self.next_page[layer][kv_head]))

    def total_valid_tokens(self, layer: int, kv_head: int) -> int:
        np = self.next_page[layer][kv_head]
        if np == 0:
            return 0
        return int(self.valid_tokens[layer, kv_head, :np].sum().item())

    def num_stored_residual_slots(self) -> Tuple[int, int]:
        """Return (used_k_slots, used_v_slots) — for debug stats."""
        total = self.k_residual_buf.shape[0]
        return (total - len(self._k_slot_free), total - len(self._v_slot_free))
