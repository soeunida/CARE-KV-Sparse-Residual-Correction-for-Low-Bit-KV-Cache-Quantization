"""baselines/turboquant_style.py — TurboQuant-style same-condition reimpl.

> **HONEST FRAMING — NOT OFFICIAL.** This is a *same-condition
> reimplementation* of the TurboQuant idea (arXiv:2504.19874, "Online
> Vector Quantization with Near-optimal Distortion Rate"). Google has
> not released official code; four community repos disagree on the QJL
> specifics (see `summaries/turboquant_integration_status.md`). We do
> NOT claim official TurboQuant numbers. Every row produced here is
> labelled `same-condition reimplementation`.

TurboQuant has three stages; we reimplement all three:

1. **Random rotation** of K/V along head_dim. Data-oblivious (online).
   We use a *seeded random orthonormal* matrix `R` (QR of a Gaussian),
   which — unlike RotateKV's fixed Walsh-Hadamard — is the randomized
   rotation TurboQuant specifies. `R @ R.T == I`, so the rotation is
   lossless in fp; quantization noise lives in the rotated basis.
2. **Per-coordinate scalar quantization** — each rotated channel
   quantized independently (per-channel for K across tokens, per-token
   for V across channels), symmetric INT`bits`.
3. **QJL (Quantized-JL) 1-bit residual inner-product correction** — the
   distinctive TurboQuant contribution. For the rotated K residual
   `r = k_rot - k_rot_hat` we store the 1-bit sketch `sign(S r)` plus
   the scalar `‖r‖` (S is a seeded Gaussian projection, m rows). At
   attention time the inner product `⟨q, k⟩` is corrected by an
   *unbiased* estimate of `⟨q_rot, r⟩`:

       ⟨q_rot, r⟩ ≈ ‖r‖ · sqrt(π/2) · (1/m) · ⟨sign(S r), S q_rot⟩

   using the Gaussian identity E[ sign(s·u)(s·v) ] = sqrt(2/π)·⟨u,v⟩/‖u‖.
   This makes the stacked attention score q·k_hat + QJL(q·r) an unbiased
   estimator of q·k — distinct from a pure rotation+quant baseline
   (which is just `RotateKVStyleAdapter` with a random rotation).

Without stage 3 this collapses into RotateKV-with-a-random-matrix, which
is exactly why the prior turn declined a rotation-only reimpl. Stage 3
is what makes this a *TurboQuant*-style row.

V is rotation + per-token scalar quant only (QJL is a key/score-side
estimator in the paper); documented and accounted in `estimate_memory`.
"""
from __future__ import annotations
import math
import os
import types
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv,
)
from .common import KVMethodAdapter, DEVICE, fp16_kv_mb

_SQRT_HALF_PI = math.sqrt(math.pi / 2.0)


# ─────────────────────────────────────────────
# Seeded random orthonormal rotation + QJL projection (cached per (d, seed))
# ─────────────────────────────────────────────
_ROT_CACHE: dict = {}
_QJL_CACHE: dict = {}


def _random_rotation(d: int, device, seed: int = 0, dtype=torch.float32) -> Tensor:
    """Seeded random orthonormal d×d matrix R (R @ R.T == I) via QR of a
    Gaussian. Data-oblivious — built once, shared across the whole cache."""
    key = (d, str(device), seed, dtype)
    R = _ROT_CACHE.get(key)
    if R is None:
        g = torch.Generator(device="cpu").manual_seed(1234 + seed)
        A = torch.randn(d, d, generator=g, dtype=torch.float32)
        Q, Rm = torch.linalg.qr(A)
        # Fix sign ambiguity so the rotation is deterministic.
        Q = Q * torch.sign(torch.diagonal(Rm)).unsqueeze(0)
        R = Q.to(device=device, dtype=dtype).contiguous()
        _ROT_CACHE[key] = R
    return R


def _qjl_matrix(d: int, m: int, device, seed: int = 0, dtype=torch.float32) -> Tensor:
    """Seeded Gaussian QJL projection S of shape (m, d)."""
    key = (d, m, str(device), seed, dtype)
    S = _QJL_CACHE.get(key)
    if S is None:
        g = torch.Generator(device="cpu").manual_seed(9876 + seed)
        S = torch.randn(m, d, generator=g, dtype=torch.float32).to(
            device=device, dtype=dtype).contiguous()
        _QJL_CACHE[key] = S
    return S


def _per_coord_quant(x_rot: Tensor, bits: int, dim: int) -> Tensor:
    """Symmetric per-coordinate scalar quant of a rotated tensor.
    `dim` is the axis the scale is shared across (reduced over):
      K: scale per channel across tokens  -> reduce over token axis (-2)
      V: scale per token across channels  -> reduce over channel axis (-1)
    Returns x_rot_hat (same shape/dtype)."""
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-(2 ** (bits - 1)))
    scale = x_rot.abs().amax(dim=dim, keepdim=True) / max(qmax, 1.0)
    scale = scale.clamp(min=1e-8)
    codes = (x_rot / scale).round().clamp(qmin, qmax)
    return codes * scale


def _turbo_forward_factory(bits_k: int, bits_v: int, qjl_m: int,
                            use_qjl: bool, rot_seed: int = 0):
    """Build a TurboQuant eager forward bound to a LlamaAttention module.

    Mirrors transformers 4.45.2 `LlamaAttention.forward` exactly, inserting
    rotation+per-coord-quant on K/V and the QJL residual score correction.
    """

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # (post-RoPE) query_states: (B,Hq,T,d)  key_states/value_states: (B,Hkv,T,d)

        d = self.head_dim
        R = _random_rotation(d, key_states.device, rot_seed, torch.float32)

        # ---- Stage 1+2: rotate, per-coordinate scalar quant, un-rotate ----
        k32 = key_states.to(torch.float32)
        k_rot = k32 @ R                                   # rotate along head_dim
        k_rot_hat = _per_coord_quant(k_rot, bits_k, dim=-2)   # per-channel across T
        k_hat = (k_rot_hat @ R.transpose(0, 1)).to(key_states.dtype)

        v32 = value_states.to(torch.float32)
        v_rot = v32 @ R
        v_rot_hat = _per_coord_quant(v_rot, bits_v, dim=-1)   # per-token across channels
        v_hat = (v_rot_hat @ R.transpose(0, 1)).to(value_states.dtype)

        # ---- Stage 3: QJL 1-bit residual inner-product correction ----
        qjl_corr = None
        if use_qjl:
            S = _qjl_matrix(d, qjl_m, key_states.device, rot_seed, torch.float32)  # (m,d)
            r = k_rot - k_rot_hat                          # rotated residual (B,Hkv,T,d)
            sign_Sr = torch.sign(r @ S.transpose(0, 1))    # (B,Hkv,T,m) in {-1,0,+1}
            r_norm = r.norm(dim=-1)                         # (B,Hkv,T)
            q_rot = query_states.to(torch.float32) @ R      # (B,Hq,T,d)
            Sq = q_rot @ S.transpose(0, 1)                  # (B,Hq,T,m)
            # repeat kv-indexed tensors up to Hq. sign_Sr is (B,Hkv,T,m) —
            # exactly repeat_kv's expected (B,Hkv,T,headdim) layout.
            sign_Sr = repeat_kv(sign_Sr, self.num_key_value_groups)
            r_norm_rep = repeat_kv(r_norm.unsqueeze(-1), self.num_key_value_groups).squeeze(-1)
            # corr[b,h,i,j] = sum_m Sq[b,h,i,m]*sign_Sr[b,h,j,m]
            corr = torch.matmul(Sq, sign_Sr.transpose(2, 3))           # (B,Hq,Tq,Tk)
            qjl_corr = corr * (_SQRT_HALF_PI / float(qjl_m)) * r_norm_rep.unsqueeze(2)

        key_hat_rep = repeat_kv(k_hat, self.num_key_value_groups)
        value_hat_rep = repeat_kv(v_hat, self.num_key_value_groups)

        # Attention backend (default eager). TURBOQUANT_ATTENTION_BACKEND=sdpa_reconstruct
        # runs F.scaled_dot_product_attention on the reconstructed K̂/V̂, folding
        # the QJL pre-softmax logit correction into SDPA's additive attn_mask
        # (causal + qjl_corr/√d) — numerically equivalent. NOTE: a full
        # (B,Hq,Tq,Tk) bias forces SDPA's math backend (no FlashAttention
        # speedup) whenever QJL is on; the flash path only triggers with QJL off.
        backend = os.environ.get("TURBOQUANT_ATTENTION_BACKEND", "eager").lower()
        scale = 1.0 / math.sqrt(self.head_dim)
        causal_mask = (attention_mask[:, :, :, : key_hat_rep.shape[-2]]
                       if attention_mask is not None else None)

        if backend == "sdpa_reconstruct":
            bias = None
            if causal_mask is not None:
                bias = causal_mask.to(query_states.dtype)
            if qjl_corr is not None:
                qb = (qjl_corr * scale).to(query_states.dtype)
                bias = qb if bias is None else bias + qb
            attn_output = F.scaled_dot_product_attention(
                query_states, key_hat_rep, value_hat_rep,
                attn_mask=bias, dropout_p=0.0, scale=scale)
            attn_weights = None
        else:  # eager (default, paper path)
            attn_weights = torch.matmul(query_states, key_hat_rep.transpose(2, 3))
            if qjl_corr is not None:
                attn_weights = attn_weights + qjl_corr.to(attn_weights.dtype)
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            if causal_mask is not None:
                attn_weights = attn_weights + causal_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_hat_rep)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    return forward


def patch_turboquant_style(model, bits_k: int, bits_v: int, qjl_m: int,
                            use_qjl: bool, rot_seed: int = 0):
    """Replace every LlamaAttention.forward with the TurboQuant forward.
    Model must be loaded with attn_implementation='eager'. Returns unpatch()."""
    from transformers.models.llama.modeling_llama import LlamaAttention
    fwd = _turbo_forward_factory(bits_k, bits_v, qjl_m, use_qjl, rot_seed)
    originals = []
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            originals.append((module, module.forward))
            module.forward = types.MethodType(fwd, module)
    if not originals:
        raise RuntimeError("No LlamaAttention modules found — load model with "
                           "attn_implementation='eager'.")

    def unpatch():
        for module, orig in originals:
            module.forward = orig
    return unpatch


class TurboQuantStyleAdapter(KVMethodAdapter):
    family = "turboquant_style"
    is_official = False
    is_reimplementation = True
    is_unsupported = False
    uses_residual = True            # QJL 1-bit residual sketch
    uses_query_aware_routing = False

    def __init__(self, bits_k: int = 3, bits_v: int = 3,
                 qjl_m: int = 0, use_qjl: bool = True, rot_seed: int = 0):
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.qjl_m = qjl_m          # 0 → default head_dim (set at setup)
        self.use_qjl = use_qjl
        self.rot_seed = rot_seed
        tag = "" if use_qjl else "_noQJL"
        self.name = f"TurboQuant_style_INT{bits_k}{tag}"
        self.bit_width = f"K=INT{bits_k}, V=INT{bits_v}" + (" + QJL 1-bit residual" if use_qjl else "")
        self.k_quant_scheme = (f"random orthonormal rotation + per-channel INT{bits_k}"
                                + (" + QJL 1-bit residual inner-product correction" if use_qjl else ""))
        self.v_quant_scheme = f"random orthonormal rotation + per-token INT{bits_v}"
        self._unpatch = None
        self._eff_m = qjl_m

    def setup_model(self, model_id: str):
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            attn_implementation="eager",
            device_map=DEVICE if DEVICE == "cuda" else None,
        )
        m.config.use_cache = False
        m.eval()
        hd = m.config.hidden_size // m.config.num_attention_heads
        # QJL break-even is m ~ 2*head_dim (below that the 1-bit estimator's
        # variance exceeds the post-quant residual energy). Default to 2*hd.
        self._eff_m = self.qjl_m if self.qjl_m > 0 else 2 * hd
        self._unpatch = patch_turboquant_style(
            m, self.bits_k, self.bits_v, self._eff_m, self.use_qjl, self.rot_seed)
        return m

    def teardown(self):
        if self._unpatch is not None:
            self._unpatch()
            self._unpatch = None

    def estimate_memory(self, seq_len: int, num_layers: int = 22,
                         hkv: int = 4, head_dim: int = 64):
        m = self.qjl_m if self.qjl_m > 0 else 2 * head_dim
        # per-coordinate scalar quant: per-channel K + per-token V + fp16 scales
        k_bytes = (seq_len * hkv * head_dim * self.bits_k / 8.0 + 2 * hkv * head_dim)
        v_bytes = (seq_len * hkv * head_dim * self.bits_v / 8.0 + 2 * hkv * seq_len)
        # QJL sketch: m bits per key + one fp16 norm per key (per kv head)
        qjl_bytes = (seq_len * hkv * m / 8.0 + 2 * hkv * seq_len) if self.use_qjl else 0.0
        base_mb = num_layers * (k_bytes + v_bytes) / (1024 * 1024)
        qjl_mb = num_layers * qjl_bytes / (1024 * 1024)
        total_mb = base_mb + qjl_mb
        fp16 = fp16_kv_mb(seq_len, num_layers, hkv, head_dim)
        return dict(
            estimated_kv_memory_MB=round(total_mb, 4),
            estimated_total_cache_memory_MB=round(total_mb, 4),
            vs_fp16_kv_memory_ratio=round(total_mb / max(fp16, 1e-9), 4),
            base_memory_MB=round(base_mb, 4),
            residual_memory_MB=round(qjl_mb, 4),
            base_quantizer="turboquant_style",
        )

    def notes(self) -> str:
        q = (f"+ QJL 1-bit residual inner-product correction (m={self._eff_m} "
             f"Gaussian projections, unbiased estimator of q·r)") if self.use_qjl else \
            "(QJL disabled — rotation+quant only, i.e. random-rotation RotateKV)"
        return (f"TurboQuant-style same-condition reimplementation (NOT official; "
                f"no official code released — see turboquant_integration_status.md). "
                f"Seeded random orthonormal rotation along head_dim + per-channel "
                f"K INT{self.bits_k} / per-token V INT{self.bits_v} {q}.")


# ─────────────────────────────────────────────
# Numerical self-test of the QJL estimator (run before trusting the eval)
# ─────────────────────────────────────────────
def selftest_qjl(d: int = 64, m: int = 64, bits: int = 3,
                 n_keys: int = 4096, n_trials: int = 1, device: str = "cpu"):
    """Verify the QJL residual estimator is ~unbiased and reduces inner-product
    MSE vs the rotated-quant base alone. Returns a dict of diagnostics."""
    torch.manual_seed(0)
    R = _random_rotation(d, device, 0)
    S = _qjl_matrix(d, m, device, 0)
    q = torch.randn(n_keys, d, device=device)
    k = torch.randn(n_keys, d, device=device)
    k_rot = k @ R
    k_rot_hat = _per_coord_quant(k_rot.unsqueeze(0).unsqueeze(0), bits, dim=-2).squeeze()
    r = k_rot - k_rot_hat
    q_rot = q @ R
    true_qk = (q * k).sum(-1)                                   # exact q·k
    base_qk = (q_rot * k_rot_hat).sum(-1)                       # q·k_hat (rotated == original)
    true_qr = (q_rot * r).sum(-1)                               # exact q·r residual
    sign_Sr = torch.sign(r @ S.transpose(0, 1))
    Sq = q_rot @ S.transpose(0, 1)
    est_qr = (Sq * sign_Sr).sum(-1) * (_SQRT_HALF_PI / float(m)) * r.norm(dim=-1)
    mse_base = ((base_qk - true_qk) ** 2).mean().item()
    mse_qjl = ((base_qk + est_qr - true_qk) ** 2).mean().item()
    return dict(
        d=d, m=m, bits=bits, n_keys=n_keys,
        mean_true_qr=true_qr.mean().item(),
        mean_est_qr=est_qr.mean().item(),
        bias=(est_qr - true_qr).mean().item(),
        corr_est_true=torch.corrcoef(torch.stack([est_qr, true_qr]))[0, 1].item(),
        mse_inner_base=mse_base,
        mse_inner_qjl=mse_qjl,
        mse_reduction_pct=100.0 * (mse_base - mse_qjl) / max(mse_base, 1e-12),
    )


if __name__ == "__main__":
    import json
    for bits in (4, 3, 2):
        print(f"--- QJL self-test bits={bits} ---")
        print(json.dumps(selftest_qjl(bits=bits), indent=2))
