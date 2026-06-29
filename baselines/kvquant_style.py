"""baselines/kvquant_style.py — KVQuant-style same-condition reimplementation.

KVQuant's distinguishing traits (per the paper):
 1. **Pre-RoPE K storage** — quantize K BEFORE rotary so the quantized
    distribution is the unrotated one (smoother per-channel).
 2. Per-channel K + per-token V quantization (similar to KIVI).
 3. Non-uniform quantization (NUQ) calibrated from data.
 4. Dense + sparse decomposition for outliers.

This adapter implements (1) + (2) as a **minimal viable**
reimplementation. NUQ and dense+sparse outlier handling are NOT
included — they would require calibration data and a sparse storage
path that's out of scope for this turn.

Honest framing:
 - "KVQuant-style same-condition reimplementation" — pre-RoPE K storage
   choice + per-channel/per-token uniform INT3 quant.
 - NOT the official KVQuant (no NUQ, no sparse outlier path, no CUDA
   kernels from `SqueezeAILab/KVQuant`).

Patch point:
 - `pre_rope`  : monkey-patch `LlamaAttention.k_proj.forward` so K is
   quantized BEFORE `apply_rotary_pos_emb` sees it. Downstream attention
   then operates on rotated_dequant(quant(K)). V handled the same way
   as KIVI (per-token V quant on v_proj output).
 - `post_rope` : fallback — monkey-patch `apply_rotary_pos_emb` like
   KIVI does, but using KVQuant's per-channel K. Use this when the
   downstream cache layout requires post-RoPE K (e.g., CARE-KV's
   cache). Labelled as "KVQuant-style (post-RoPE variant)".

Stacking with CARE-KV:
 - CARE-KV's cache stores **post-RoPE K**. True pre-RoPE KVQuant is
   incompatible with that cache layout (would need a new pre-RoPE K
   storage path through cache.py + layer.py + prefill loop — same
   blocker as the original `KVQuantStyleAdapter` STUB documented).
 - For the WT-2 N=4 SL=128 pilot, **standalone KVQuant_style_INT3 runs
   in pre_rope mode**, and **KVQuant_style + CARE-KV is recorded as
   unsupported with the blocker**. A future turn can either (a) add the
   pre-RoPE K cache path, or (b) ship a "KVQuant-style post-RoPE
   variant + CARE-KV" cell that uses the same kivi_style dispatch in
   CARE-KV's cache (Phase Q-stacked side-buffer). Option (b) is
   straightforward — same code path as KIVI-style + CARE-KV, just with
   the per-channel K computed on the pre-rotary signal then rotated
   back in. Out of scope for this turn.
"""
from __future__ import annotations
from typing import Optional

import torch
from torch import Tensor

from transformers import LlamaForCausalLM
from .common import KVMethodAdapter, DEVICE, fp16_kv_mb


def _per_channel_K_quant_dequant(K: Tensor, bits: int) -> Tensor:
    """K shape: (B, T, Hkv*D) OR (B, Hkv, T, D). Per-channel symmetric INT
    quant, scale taken across the token dimension per (B, Hkv, channel).
    Returns same shape.
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    if K.dim() == 3:
        # (B, T, Hkv*D) — scale per (B, channel) across T
        scale = K.abs().amax(dim=1, keepdim=True) / qmax
    elif K.dim() == 4:
        # (B, Hkv, T, D) — scale per (B, Hkv, channel) across T
        scale = K.abs().amax(dim=2, keepdim=True) / qmax
    else:
        raise ValueError(f"unexpected K shape {K.shape}")
    scale = scale.clamp(min=1e-8)
    codes = (K / scale).round().clamp(qmin, qmax)
    return codes * scale


def _per_token_V_quant_dequant(V: Tensor, bits: int) -> Tensor:
    """V shape: (B, T, Hkv*D) OR (B, Hkv, T, D). Per-token symmetric INT
    quant, scale taken across the channel dimension per (B, Hkv, token).
    """
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    if V.dim() == 3:
        # (B, T, Hkv*D) — scale per (B, T)  across Hkv*D
        scale = V.abs().amax(dim=2, keepdim=True) / qmax
    elif V.dim() == 4:
        # (B, Hkv, T, D) — scale per (B, Hkv, T) across D
        scale = V.abs().amax(dim=3, keepdim=True) / qmax
    else:
        raise ValueError(f"unexpected V shape {V.shape}")
    scale = scale.clamp(min=1e-8)
    codes = (V / scale).round().clamp(qmin, qmax)
    return codes * scale


def patch_kvquant_style(model, bits_k: int, bits_v: int,
                          k_store_mode: str = "pre_rope"):
    """Monkey-patch K projection (pre-RoPE) or apply_rotary_pos_emb
    (post-RoPE fallback) + v_proj.

    Returns an `unpatch` callable that reverses the rotary monkey-patch
    (k_proj/v_proj are restored by re-running setup_model)."""
    from transformers.models.llama import modeling_llama

    orig_apply_rotary = modeling_llama.apply_rotary_pos_emb

    if k_store_mode == "post_rope":
        def wrapped_apply_rotary(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            q_out, k_out = orig_apply_rotary(q, k, cos, sin,
                                               position_ids=position_ids,
                                               unsqueeze_dim=unsqueeze_dim)
            # k_out: (B, Hkv, T, D). KVQuant per-channel K, post-RoPE variant.
            k_hat = _per_channel_K_quant_dequant(k_out, bits_k)
            return q_out, k_hat
        modeling_llama.apply_rotary_pos_emb = wrapped_apply_rotary
        k_patched_pre = False
    elif k_store_mode == "pre_rope":
        # No rotary patch — K is quantized BEFORE rotary on k_proj output.
        k_patched_pre = True
    else:
        raise ValueError(f"k_store_mode must be pre_rope or post_rope, got {k_store_mode}")

    cfg_model = model.config
    hkv = cfg_model.num_key_value_heads
    d = cfg_model.hidden_size // cfg_model.num_attention_heads

    # V: always wrap v_proj (per-token V is the same whether KVQuant or KIVI).
    def v_proj_wrap(v_proj_module):
        orig_forward = v_proj_module.forward
        def new_forward(x):
            v_lin = orig_forward(x)
            B, T, HD = v_lin.shape
            v = v_lin.view(B, T, hkv, d).transpose(1, 2)
            v_hat = _per_token_V_quant_dequant(v, bits_v)
            return v_hat.transpose(1, 2).reshape(B, T, HD)
        v_proj_module.forward = new_forward

    # K: wrap k_proj only in pre_rope mode.
    def k_proj_wrap(k_proj_module):
        orig_forward = k_proj_module.forward
        def new_forward(x):
            k_lin = orig_forward(x)
            B, T, HD = k_lin.shape
            k = k_lin.view(B, T, hkv, d).transpose(1, 2)   # (B, Hkv, T, D)
            k_hat = _per_channel_K_quant_dequant(k, bits_k)
            return k_hat.transpose(1, 2).reshape(B, T, HD)
        k_proj_module.forward = new_forward

    for name, module in model.named_modules():
        if hasattr(module, "v_proj") and hasattr(module.v_proj, "forward"):
            v_proj_wrap(module.v_proj)
        if k_patched_pre and hasattr(module, "k_proj") and hasattr(module.k_proj, "forward"):
            k_proj_wrap(module.k_proj)

    def unpatch():
        if not k_patched_pre:
            modeling_llama.apply_rotary_pos_emb = orig_apply_rotary
    return unpatch


class KVQuantStyleAdapter(KVMethodAdapter):
    family = "kvquant_style"
    is_official = False
    is_reimplementation = True
    is_unsupported = False
    uses_residual = False
    uses_mixed_precision = False
    uses_token_eviction = False
    uses_query_aware_routing = False

    def __init__(self, bits_k: int = 3, bits_v: int = 3,
                  k_store_mode: str = "pre_rope"):
        if k_store_mode not in ("pre_rope", "post_rope"):
            raise ValueError(f"k_store_mode must be pre_rope or post_rope, got {k_store_mode}")
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.k_store_mode = k_store_mode
        tag = "preRoPE" if k_store_mode == "pre_rope" else "postRoPE"
        self.name = f"KVQuant_style_INT{bits_k}K_INT{bits_v}V_{tag}"
        self.bit_width = f"K=INT{bits_k} ({tag}), V=INT{bits_v} (per-token)"
        self.k_quant_scheme = (
            f"per-channel (scale across tokens), INT{bits_k} symmetric, "
            f"{'PRE-RoPE storage' if k_store_mode == 'pre_rope' else 'POST-RoPE storage (fallback)'}"
        )
        self.v_quant_scheme = f"per-token (scale across channels), INT{bits_v} symmetric"
        self._unpatch = None

    def setup_model(self, model_id: str):
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map=DEVICE if DEVICE == "cuda" else None,
        )
        m.config.use_cache = False
        m.eval()
        self._unpatch = patch_kvquant_style(m, self.bits_k, self.bits_v,
                                              self.k_store_mode)
        return m

    def teardown(self):
        if self._unpatch is not None:
            self._unpatch()
            self._unpatch = None

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                          hkv: int = 4, head_dim: int = 64):
        # Same theoretical footprint as KIVI: per-channel K bits + per-token V bits + fp16 scale headers.
        # The pre-RoPE vs post-RoPE choice doesn't change memory.
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
            base_quantizer="kvquant_style",
        )

    def notes(self) -> str:
        return (f"KVQuant-style same-condition reimplementation. "
                f"Per-channel K INT{self.bits_k} ({self.k_store_mode}) + "
                f"per-token V INT{self.bits_v}. Dense uniform quant only — "
                f"NUQ and dense+sparse outlier handling NOT included. "
                f"NOT the official KVQuant CUDA repo "
                f"(SqueezeAILab/KVQuant); calibration-free.")


class KVQuantPlusCAREKVUnsupported(KVMethodAdapter):
    """Placeholder for KVQuant + CARE-KV stacked.

    Reason: true pre-RoPE KVQuant requires storing K BEFORE rotary in the
    cache. CARE-KV's cache stores **post-RoPE K** (validated by the
    Phase K-c activation diagnostics + the READ=0 invariant). Pre-RoPE
    storage needs a new K-store-mode switch through cache.py +
    layer.py + the prefill loop — same blocker as the previous
    KVQuantStyleAdapter STUB.

    A post-RoPE variant of KVQuant-style is the same as KIVI-style for
    the K side; that cell already exists (KIVI_INT3 + CARE-KV) in
    Phase Q-stacked. So adding a "KVQuant (post-RoPE) + CARE-KV"
    cell would be a relabel, not new information.
    """
    family = "kvquant_plus_carekv"
    is_official = False
    is_reimplementation = False
    is_unsupported = True
    bit_width = "K=INT3 (pre-RoPE) + CARE-KV residual"
    k_quant_scheme = "per-channel (pre-RoPE) + CARE-KV residual"
    v_quant_scheme = "per-token + CARE-KV residual"
    uses_residual = True
    uses_query_aware_routing = True
    unsupported_reason = (
        "True pre-RoPE KVQuant + CARE-KV requires a new K-store-mode "
        "switch through cache.py + layer.py + the prefill loop "
        "(CARE-KV's cache stores POST-RoPE K). Estimated 1–2 days. "
        "A post-RoPE KVQuant variant + CARE-KV would be the same code "
        "path as KIVI + CARE-KV (Phase Q-stacked) — see "
        "`KIVI_INT3K_INT3V_plus_CAREKV` row in "
        "`ablations/carekv_on_base_quantizers.csv`."
    )

    def __init__(self, bits: int = 3):
        self.bits = bits
        self.name = f"KVQuant_style_INT{bits}_plus_CAREKV"
        self.bit_width = f"K=INT{bits} (pre-RoPE) + CARE-KV residual"

    def setup_model(self, model_id: str):
        raise NotImplementedError(self.unsupported_reason)

    def notes(self) -> str:
        return self.unsupported_reason
