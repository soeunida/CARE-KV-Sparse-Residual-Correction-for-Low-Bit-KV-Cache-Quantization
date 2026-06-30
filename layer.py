"""
care_kv/layer.py
-----------------
CARE-KV layer wrapper.

Debug stats:
    When CAREKV_DEBUG_STATS=1, every carekv_stored prefill / decode step
    accumulates per-query statistics into a process-global dict accessible
    via get_debug_stats() / reset_debug_stats() below.  No-op when disabled.

Modes (env: CAREKV_PREFILL_MODE):
  fp             — full precision K/V for the output (upper bound).
  base_quant     — quantize→dequantize K/V (post-RoPE for K), attention on K_hat/V_hat.
  carekv_eval    — base_quant + sparse correction using FULL in-memory R_K/R_V
                   (upper bound for the correction policy, not slot-faithful).
  carekv_stored  — base_quant + sparse correction reading ONLY stored slots.
                   Use this for paper PPL.
  carekv         — alias for carekv_eval (backward compatibility; deprecated).

Cache layout (KV-head indexed, post-RoPE K):
  base_K_codes[L, Hkv, P, T, D] stores post-RoPE K.
  base_V_codes[L, Hkv, P, T, D] stores V.
  valid_tokens[L, Hkv, P] records per-page valid count.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch import Tensor

from .attention import (
    CAREKVMultiHeadAttention, apply_slot_corrections,
    vectorized_v_correction, vectorized_joint_correction,
)
from .cache import CAREKVCache, CacheConfig
from .quantizer import QuantConfig, dequantize, quantize_and_residual
from .residual_router import ResidualRouter
from .residual_store import ResidualStoreManager

# Phase Q-stacked + base-quantizer-expansion: dispatch helpers for the
# kivi_style / rotatekv_style / kvquant_style (post-RoPE variant) base
# quantizers. All three share the fp16-side-buffer cache layout added
# in Phase Q. The dispatch is the only difference between them.
from .kivi_helpers import (
    quant_dequant_kivi_k as _kivi_quant_dequant_K,
    quant_dequant_kivi_v as _kivi_quant_dequant_V,
    dispatch_base_kv_quant as _dispatch_base_kv_quant,
)

# Base-quantizer names that use the fp16 side-buffer dispatch (i.e.,
# any non-"uniform" name). Update this set when adding new schemes.
_SIDE_BUFFER_BASE_QUANTIZERS = {"kivi_style", "rotatekv_style",
                                "randrot_style", "kvquant_style"}
# Rotation base quantizers that support a pre-RoPE store mode (rotate+quant in
# pre-RoPE coords, then re-apply RoPE to K_hat so the residual stays post-RoPE).
_ROTATION_BASE_QUANTIZERS = {"rotatekv_style", "randrot_style"}

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
except Exception:  # pragma: no cover
    apply_rotary_pos_emb = None
    repeat_kv = None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _repeat_kv_local(x: Tensor, num_key_value_groups: int) -> Tensor:
    """Fallback repeat_kv.  x: (B, Hkv, T, D) → (B, Hkv*g, T, D)."""
    if repeat_kv is not None:
        return repeat_kv(x, num_key_value_groups)
    if num_key_value_groups == 1:
        return x
    bsz, num_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_key_value_groups, seq_len, head_dim
    )
    return x.reshape(bsz, num_kv_heads * num_key_value_groups, seq_len, head_dim)


# ─────────────────────────────────────────────
# Process-global debug stats (enabled by CAREKV_DEBUG_STATS=1)
# ─────────────────────────────────────────────

_DEBUG_STATS: Dict[str, Any] = {
    "v_slots_read": 0, "k_slots_read": 0, "n_queries": 0,
    "delta_v_norm_sum": 0.0, "delta_k_norm_sum": 0.0,
    "delta_o_norm_sum": 0.0, "o_base_norm_sum": 0.0,
}

def _debug_stats_enabled() -> bool:
    return os.environ.get("CAREKV_DEBUG_STATS", "0") == "1"

def get_debug_stats() -> Dict[str, Any]:
    """Return a snapshot of the global debug stats dict."""
    return dict(_DEBUG_STATS)

def reset_debug_stats() -> None:
    for k in list(_DEBUG_STATS.keys()):
        _DEBUG_STATS[k] = 0 if isinstance(_DEBUG_STATS[k], int) else 0.0


def _resolve_prefill_mode() -> str:
    raw = os.environ.get("CAREKV_PREFILL_MODE", "fp").lower()
    if raw == "carekv":
        warnings.warn(
            "CAREKV_PREFILL_MODE=carekv is deprecated and aliases to carekv_eval. "
            "Use carekv_stored for paper-quality slot-faithful results.",
            DeprecationWarning, stacklevel=2,
        )
        return "carekv_eval"
    if raw in {"quant", "int"}:
        return "base_quant"
    return raw


# ─────────────────────────────────────────────
# CARE-KV layer
# ─────────────────────────────────────────────

class CAREKVLayer(nn.Module):
    """
    Drop-in attention layer.  Takes Q/K/V/O projection weights and runs
    prefill / decode through the CARE-KV cache.
    """

    def __init__(
        self,
        cfg: CacheConfig,
        layer_id: int,
        W_Q: Tensor,
        W_K: Tensor,
        W_V: Tensor,
        W_O: Tensor,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.device = device

        self.W_Q = nn.Parameter(W_Q, requires_grad=False)
        self.W_K = nn.Parameter(W_K, requires_grad=False)
        self.W_V = nn.Parameter(W_V, requires_grad=False)
        self.W_O = nn.Parameter(W_O, requires_grad=False)

        self.base_qcfg = QuantConfig(bits=cfg.base_bits, group_size=cfg.group_size)

        # One store manager + one router per layer; both are stateless w.r.t.
        # KV head (they take kv_head as an argument).
        self.store_manager = ResidualStoreManager(cfg, layer_id, device)
        self.router = ResidualRouter(cfg, layer_id, device)
        self.attn = CAREKVMultiHeadAttention(cfg, layer_id, device)

    # ─────────────────────────────────────────
    # Internal utilities
    # ─────────────────────────────────────────

    def _quantize_dequantize(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return (codes, scale, x_hat) for a (T, D) tensor.

        This is the **uniform** per-group quantization path. For
        cfg.base_quantizer == 'kivi_style' the caller should compute
        K_hat / V_hat via _kivi_quant_dequant_K / _V on the full
        sequence and stash them through cache.write_base_kivi(...).
        """
        codes, scale, _, _ = quantize_and_residual(x, self.base_qcfg)
        x_hat = dequantize(codes, scale, x.shape, self.base_qcfg).to(x.dtype)
        return codes, scale, x_hat

    def _kivi_bits(self) -> Tuple[int, int]:
        """Return (k_bits, v_bits) honoring per-kind overrides."""
        cfg = self.cfg
        k_bits = cfg.k_bits_override if getattr(cfg, "k_bits_override", -1) > 0 else cfg.base_bits
        v_bits = cfg.v_bits_override if getattr(cfg, "v_bits_override", -1) > 0 else cfg.base_bits
        return int(k_bits), int(v_bits)

    def _write_pages_for_kv_head(
        self,
        cache: CAREKVCache,
        kv_h: int,
        K_kv: Tensor,           # (T, D) post-RoPE
        V_kv: Tensor,           # (T, D)
        token_start_abs: int,
        sink_tokens: int,
        K_hat_seq_override: Optional[Tensor] = None,   # (T, D) precomputed post-RoPE K_hat
        V_hat_seq_override: Optional[Tensor] = None,   # (T, D) precomputed V_hat
    ) -> Tuple[Tensor, Tensor]:
        """
        Quantize, store per-page base codes, run residual store manager.
        Returns (K_hat_concat, V_hat_concat) of shape (T, D) — the dequantized
        post-RoPE K/V actually committed to the cache, useful for letting the
        caller compute base attention without re-reading.
        """
        cfg = self.cfg
        T, D = K_kv.shape
        page_size = cfg.page_size
        num_pages = math.ceil(T / page_size)

        # Phase Q-stacked + base-quantizer-expansion: when
        # cfg.base_quantizer != "uniform", precompute K_hat / V_hat over
        # the FULL sequence (per-channel K scale is meaningfully
        # computed only across many tokens), then slice per-page. The
        # dispatch covers kivi_style / rotatekv_style / kvquant_style
        # — all three share the same fp16 side-buffer cache layout.
        bq_name = getattr(cfg, "base_quantizer", "uniform")
        use_kivi = bq_name in _SIDE_BUFFER_BASE_QUANTIZERS
        if use_kivi:
            if K_hat_seq_override is not None and V_hat_seq_override is not None:
                # KVQuant pre-RoPE path: K_hat / V_hat were precomputed by the
                # caller (prefill) in pre-RoPE coordinates and then rotated, so
                # the residual K_orig(post-RoPE) - K_hat(post-RoPE) lands in the
                # post-RoPE coordinate system the correction reads.
                K_hat_seq = K_hat_seq_override.to(K_kv.dtype)   # (T, D)
                V_hat_seq = V_hat_seq_override.to(V_kv.dtype)   # (T, D)
            else:
                k_bits, v_bits = self._kivi_bits()
                K_hat_seq, V_hat_seq = _dispatch_base_kv_quant(
                    bq_name, K_kv, V_kv, k_bits, v_bits,
                )
                K_hat_seq = K_hat_seq.to(K_kv.dtype)   # (T, D)
                V_hat_seq = V_hat_seq.to(V_kv.dtype)   # (T, D)

        K_hat_chunks = []
        V_hat_chunks = []

        for p in range(num_pages):
            t0 = p * page_size
            t1 = min(t0 + page_size, T)
            num_valid = t1 - t0

            K_page = K_kv[t0:t1]
            V_page = V_kv[t0:t1]

            if num_valid < page_size:
                pad = torch.zeros(
                    page_size - num_valid, D,
                    device=K_kv.device, dtype=K_kv.dtype,
                )
                K_page_full = torch.cat([K_page, pad], dim=0)
                V_page_full = torch.cat([V_page, pad], dim=0)
            else:
                K_page_full = K_page
                V_page_full = V_page

            page_id = cache.alloc_page(self.layer_id, kv_h)
            is_sink = (token_start_abs + t0) < sink_tokens
            is_recent = p >= num_pages - 4

            if use_kivi:
                # KIVI path: stash K_hat / V_hat in the fp16 side buffer
                # and pass them as overrides to process_page so it
                # computes R = K_orig - K_hat_kivi (not K_hat_uniform).
                K_hat_page = K_hat_seq[t0:t1]
                V_hat_page = V_hat_seq[t0:t1]
                if num_valid < page_size:
                    pad = torch.zeros(
                        page_size - num_valid, D,
                        device=K_kv.device, dtype=K_kv.dtype,
                    )
                    K_hat_full = torch.cat([K_hat_page, pad], dim=0)
                    V_hat_full = torch.cat([V_hat_page, pad], dim=0)
                else:
                    K_hat_full = K_hat_page
                    V_hat_full = V_hat_page

                cache.write_base_kivi(
                    self.layer_id, kv_h, page_id,
                    K_hat_full, V_hat_full,
                    num_valid=num_valid,
                )

                self.store_manager.process_page(
                    cache=cache, kv_head=kv_h, page_id=page_id,
                    K_orig=K_page_full, V_orig=V_page_full,
                    K_codes=None, K_scale=None,
                    V_codes=None, V_scale=None,
                    token_start=token_start_abs + t0,
                    num_valid=num_valid,
                    is_recent=is_recent, is_sink=is_sink,
                    K_hat_override=K_hat_full, V_hat_override=V_hat_full,
                )
            else:
                # Uniform path (paper-best, unchanged).
                K_codes, K_scale, K_hat_full = self._quantize_dequantize(K_page_full)
                V_codes, V_scale, V_hat_full = self._quantize_dequantize(V_page_full)

                cache.write_base(
                    self.layer_id, kv_h, page_id,
                    K_codes, K_scale, V_codes, V_scale,
                    num_valid=num_valid,
                )

                self.store_manager.process_page(
                    cache=cache, kv_head=kv_h, page_id=page_id,
                    K_orig=K_page_full, V_orig=V_page_full,
                    K_codes=K_codes, K_scale=K_scale,
                    V_codes=V_codes, V_scale=V_scale,
                    token_start=token_start_abs + t0,
                    num_valid=num_valid,
                    is_recent=is_recent, is_sink=is_sink,
                )

            K_hat_chunks.append(K_hat_full[:num_valid])
            V_hat_chunks.append(V_hat_full[:num_valid])

        return torch.cat(K_hat_chunks, dim=0), torch.cat(V_hat_chunks, dim=0)

    def _lowrank_correct_eval(self, K_post, V, K_hat, V_hat, r, kind):
        """Eval-mode rank-r dense correction (UPPER BOUND). Per kv_head, project
        the residual R = X − X̂ onto its top-r right singular vectors P (channel
        subspace) and add it back: X̂' = X̂ + (R P) Pᵀ. Exact (un-quantized)
        residual SVD → best achievable rank-r correction. r=0 ⇒ identity;
        r→D ⇒ exact. X ∈ {K_post, V}, all (Hkv, T, D)."""
        def _correct(X, Xh):
            Xc = Xh.clone()
            for h in range(X.shape[0]):
                R = (X[h] - Xh[h]).float()                 # (T, D)
                rr = min(r, R.shape[0], R.shape[1])
                if rr <= 0:
                    continue
                _, _, Vh = torch.linalg.svd(R, full_matrices=False)
                P = Vh[:rr].T                              # (D, rr)
                Xc[h] = Xh[h] + ((R @ P) @ P.T).to(Xh.dtype)
            return Xc
        if kind in ("k", "both"):
            K_hat = _correct(K_post, K_hat)
        if kind in ("v", "both"):
            V_hat = _correct(V, V_hat)
        return K_hat, V_hat

    # ─────────────────────────────────────────
    # Eval-mode prefill correction (in-memory residuals)
    # ─────────────────────────────────────────

    def _apply_sparse_prefill_correction_eval(
        self,
        o_base: Tensor,     # (1, Hq, T, D)
        attn_w: Tensor,     # (1, Hq, T, T)
        q_b: Tensor,        # (1, Hq, T, D)  post-RoPE query
        v_base: Tensor,     # (1, Hq, T, D)
        r_k: Tensor,        # (1, Hq, T, D)
        r_v: Tensor,        # (1, Hq, T, D)
    ) -> Tensor:
        """Per-token slot-free correction using full residuals (upper bound)."""
        _, Hq, T, D = q_b.shape

        residual_ratio = float(
            os.environ.get("CAREKV_PREFILL_RESIDUAL_RATIO", str(self.cfg.read_budget_ratio))
        )
        min_residuals = int(os.environ.get("CAREKV_PREFILL_MIN_RESIDUALS", "1"))
        k_corr_scale = float(os.environ.get("CAREKV_K_CORRECTION_SCALE", "0.1"))

        residual_kind = os.environ.get("CAREKV_PREFILL_RESIDUAL_KIND", "both").lower()
        if residual_kind not in {"v", "k", "both"}:
            raise ValueError(f"Unknown CAREKV_PREFILL_RESIDUAL_KIND={residual_kind}")

        v_score_kind = os.environ.get("CAREKV_V_SCORE", "output_aware").lower()
        if v_score_kind not in {"norm", "output_aware"}:
            raise ValueError(f"Unknown CAREKV_V_SCORE={v_score_kind}")

        score_normalize = os.environ.get("CAREKV_SCORE_NORMALIZE", "0") == "1"

        out_corr = o_base.clone()
        scale_val = 1.0 / math.sqrt(D)

        for h_idx in range(Hq):
            for t_idx in range(T):
                n_valid = t_idx + 1
                topk = max(min_residuals, int(n_valid * residual_ratio))
                topk = min(topk, n_valid)
                if topk <= 0:
                    continue

                a_i = attn_w[0, h_idx, t_idx, :n_valid].float()
                q_i = q_b[0, h_idx, t_idx].float()
                v_i = v_base[0, h_idx, :n_valid].float()
                rk_i = r_k[0, h_idx, :n_valid].float()
                rv_i = r_v[0, h_idx, :n_valid].float()
                o_i = o_base[0, h_idx, t_idx].float()

                v_diff_norm = (v_i - o_i.unsqueeze(0)).norm(dim=-1)

                if v_score_kind == "norm":
                    v_score = a_i * rv_i.norm(dim=-1)
                else:
                    v_score = a_i * rv_i.norm(dim=-1) * v_diff_norm

                qdot_rk = (rk_i * q_i.unsqueeze(0)).sum(dim=-1) * scale_val
                k_score = a_i * qdot_rk.abs() * v_diff_norm

                if residual_kind == "v":
                    score = v_score
                elif residual_kind == "k":
                    score = k_score
                else:
                    if score_normalize:
                        vm = v_score.abs().mean().clamp(min=1e-8)
                        km = k_score.abs().mean().clamp(min=1e-8)
                        score = v_score / vm + k_score / km
                    else:
                        score = v_score + k_score

                idx = torch.topk(score, k=topk, largest=True).indices
                a_sel = a_i[idx]

                if residual_kind in {"v", "both"}:
                    dO_V = (a_sel.unsqueeze(-1) * rv_i[idx]).sum(dim=0)
                else:
                    dO_V = torch.zeros_like(o_i)

                if residual_kind in {"k", "both"}:
                    qdot_sel = qdot_rk[idx].clamp(-2.0, 2.0)
                    dO_K = (a_sel * qdot_sel).unsqueeze(-1) * (
                        v_i[idx] - o_i.unsqueeze(0)
                    )
                    dO_K = dO_K.sum(dim=0)
                else:
                    dO_K = torch.zeros_like(o_i)

                out_corr[0, h_idx, t_idx] = (
                    o_i + dO_V + k_corr_scale * dO_K
                ).to(out_corr.dtype)

        return out_corr

    # ─────────────────────────────────────────
    # Stored-slot prefill correction (true CARE-KV)
    # ─────────────────────────────────────────

    def _apply_sparse_prefill_correction_stored(
        self,
        cache: CAREKVCache,
        o_base: Tensor,     # (Hq, T, D)
        attn_w: Tensor,     # (Hq, T, N_total)  — already masked
        q_post: Tensor,     # (Hq, T, D)        — post-RoPE
        s_base: Tensor,     # (Hq, T, N_total)  — pre-softmax logits
        V_base_full: Tensor, # (Hkv, N_total, D) — dequant base V, KV-head indexed
        debug_stats: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Compute per-(head, t) ΔO via apply_slot_corrections reading only
        from stored cache slots.
        """
        cfg = self.cfg
        Hq, T, D = q_post.shape
        Hkv = cfg.num_kv_heads
        kv_group = Hq // Hkv

        kind = os.environ.get("CAREKV_PREFILL_RESIDUAL_KIND", "both").lower()
        if kind not in {"v", "k", "both"}:
            raise ValueError(f"Unknown CAREKV_PREFILL_RESIDUAL_KIND={kind}")
        k_corr_scale = float(os.environ.get("CAREKV_K_CORRECTION_SCALE", "0.1"))
        score_normalize = os.environ.get("CAREKV_SCORE_NORMALIZE", "0") == "1"

        out_corr = o_base.clone()

        # All page_ids for each KV head (cache currently holds exactly the
        # prefill we just wrote).
        page_ids_by_kvh = {
            kv_h: cache.all_page_ids(self.layer_id, kv_h) for kv_h in range(Hkv)
        }

        # P4-cached: per (layer, kv_head) pre-unpacked-slot cache shared
        # across all (h, t) queries.  apply_slot_corrections populates this
        # lazily on first slot access and reuses on subsequent calls.
        use_cached = (cfg.correction_impl in {"cached", "vectorized"})
        # P4-vectorized: batch V scoring + topk + accumulation over (h × T)
        # via torch ops.  Bit-equivalent to cached when V's selection is
        # independent of K's (kind=v OR policy ∈ {separate, k_first}).  For
        # policy=joint AND kind=both, the cached path interleaves K and V
        # via a normalized joint top-k that our vectorized V doesn't yet
        # reproduce, so we fall back to cached for that combination.
        joint_both = (kind == "both" and getattr(cfg, "route_policy", "separate") == "joint")
        use_vectorized = (cfg.correction_impl == "vectorized" and not joint_both)

        N_total = attn_w.shape[-1]

        # P5-full-vectorized: replace the per-(h, t) router+correction Python
        # loop with one batched call per kv_head over all (kv_group × T)
        # queries.  Handles every kind/policy (incl. joint+both) — bit-close
        # to the cached loop (≤1e-4) with identical K/V read counts.
        if cfg.correction_impl == "vectorized":
            for kv_h in range(Hkv):
                head_idxs = [h for h in range(Hq) if h // kv_group == kv_h]
                if not head_idxs:
                    continue
                page_ids = page_ids_by_kvh[kv_h]
                V_base_kvh = V_base_full[kv_h]                    # (N_total, D)
                Q_q = q_post[head_idxs].reshape(-1, D)
                S = s_base[head_idxs].reshape(-1, N_total)
                A = attn_w[head_idxs].reshape(-1, N_total)
                O = o_base[head_idxs].reshape(-1, D)
                delta = vectorized_joint_correction(
                    cache, cfg, self.router, self.layer_id, kv_h, page_ids,
                    Q_q=Q_q, S=S, A=A, V_base=V_base_kvh, O_base=O,
                    kind=kind, k_corr_scale=k_corr_scale,
                    score_normalize=score_normalize, debug_stats=debug_stats,
                ).reshape(len(head_idxs), T, D)
                for i, h in enumerate(head_idxs):
                    out_corr[h] = (o_base[h] + delta[i]).to(out_corr.dtype)
            return out_corr

        # Vectorized path: precompute ΔO_V for ALL (h, t) using batched
        # tensor ops per (kv_head).  K correction stays on the cached
        # per-(h, t) path (hybrid — kept identical to the cached impl).
        delta_V_all: Optional[Tensor] = None
        if use_vectorized and kind in {"v", "both"}:
            # Keep in fp32 so the subsequent (delta_K + delta_V) accumulation
            # matches the cached path, which is fully fp32 until the final
            # cast to out_corr.dtype.
            delta_V_all = torch.zeros(Hq, T, D, device=o_base.device,
                                      dtype=torch.float32)
            for kv_h in range(Hkv):
                page_ids = page_ids_by_kvh[kv_h]
                head_idxs = [h for h in range(Hq) if h // kv_group == kv_h]
                if not head_idxs:
                    continue
                A_concat = attn_w[head_idxs].reshape(-1, N_total).float()
                dv, _ = vectorized_v_correction(
                    cache=cache, cfg=cfg,
                    layer_id=self.layer_id, kv_head=kv_h, page_ids=page_ids,
                    A=A_concat, N_total=N_total, D=D,
                    debug_stats=debug_stats,
                )
                if dv is None:
                    continue
                dv = dv.reshape(len(head_idxs), T, D)
                for i, h in enumerate(head_idxs):
                    delta_V_all[h] = dv[i]                        # stay fp32

        # K correction (and V correction for non-vectorized paths) — keep
        # the existing per-(h, t) cached/python implementation.  When
        # vectorized V is active we still call apply_slot_corrections with
        # kind="k" so it skips the V branch.
        for h in range(Hq):
            kv_h = h // kv_group
            page_ids = page_ids_by_kvh[kv_h]
            V_base_kvh = V_base_full[kv_h]              # (N_total, D)
            kvh_slot_cache: Optional[Dict[Any, Tensor]] = ({} if use_cached else None)

            # Decide per-(h) kind for the legacy call:
            # - vectorized + kind=v   → skip legacy (already done above)
            # - vectorized + kind=k   → run legacy K
            # - vectorized + kind=both→ run legacy K only (V already done)
            # - cached/python         → run legacy normally
            legacy_kind = kind
            if use_vectorized:
                if kind == "v":
                    legacy_kind = None
                elif kind == "both":
                    legacy_kind = "k"
                # kind == "k" stays "k"

            for t in range(T):
                a_t = attn_w[h, t].float()
                s_t = s_base[h, t].float()
                o_t = o_base[h, t].float()
                q_t = q_post[h, t].float()

                if legacy_kind is not None:
                    delta_legacy = apply_slot_corrections(
                        cache=cache, cfg=cfg, router=self.router,
                        layer_id=self.layer_id, kv_head=kv_h,
                        page_ids=page_ids,
                        q=q_t, s_base=s_t, a_base=a_t,
                        V_base=V_base_kvh, O_base=o_t,
                        kind=legacy_kind, k_corr_scale=k_corr_scale,
                        score_normalize=score_normalize,
                        debug_stats=debug_stats,
                        slot_cache=kvh_slot_cache,
                    ).float()
                else:
                    delta_legacy = torch.zeros(D, device=o_base.device, dtype=torch.float32)

                # Vectorized V (if applicable)
                if delta_V_all is not None:
                    delta_legacy = delta_legacy + delta_V_all[h, t].float()

                out_corr[h, t] = (o_t + delta_legacy).to(out_corr.dtype)

        return out_corr

    # ─────────────────────────────────────────
    # Prefill
    # ─────────────────────────────────────────

    def prefill(
        self,
        cache: CAREKVCache,
        hidden: Tensor,                                 # (T, model_dim)
        sink_tokens: int = 4,
        attention_mask: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        token_start_abs: int = 0,
        debug_stats: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        cfg = self.cfg
        dev = hidden.device
        T, _ = hidden.shape

        Hq = int(cfg.num_heads)
        Hkv = int(cfg.num_kv_heads)
        D = int(cfg.head_dim)
        assert Hq % Hkv == 0
        kv_group = Hq // Hkv

        prefill_mode = _resolve_prefill_mode()

        # Project
        Q = hidden @ self.W_Q.T                         # (T, Hq*D)
        K = hidden @ self.W_K.T                         # (T, Hkv*D)
        V = hidden @ self.W_V.T                         # (T, Hkv*D)

        Q = Q.reshape(T, Hq, D).permute(1, 0, 2).contiguous()      # (Hq, T, D)
        K = K.reshape(T, Hkv, D).permute(1, 0, 2).contiguous()     # (Hkv, T, D)
        V = V.reshape(T, Hkv, D).permute(1, 0, 2).contiguous()     # (Hkv, T, D)

        # Apply RoPE (required for LLaMA-style models).
        if position_embeddings is not None and apply_rotary_pos_emb is not None:
            cos, sin = position_embeddings
            cos = cos.to(device=dev, dtype=Q.dtype)
            sin = sin.to(device=dev, dtype=Q.dtype)
            q_b = Q.unsqueeze(0)
            k_b = K.unsqueeze(0)
            q_b, k_b = apply_rotary_pos_emb(q_b, k_b, cos, sin)
            Q_post = q_b[0]          # (Hq, T, D)
            K_post = k_b[0]          # (Hkv, T, D)
            rope_applied = True
            rot = cos.shape[-1]      # rotary dimension applied by RoPE (== D for full RoPE)
        else:
            Q_post = Q
            K_post = K
            rope_applied = False
            rot = 0                  # no RoPE → pre-RoPE rotation path disabled

        # ── KVQuant pre-RoPE base-quant (true KVQuant trait) ──────────
        # When base_quantizer == "kvquant_style" and k_store_mode ==
        # "pre_rope", quantize K in PRE-RoPE coordinates (the smoother
        # unrotated per-channel distribution KVQuant exploits), then
        # re-apply RoPE to K_hat so the residual R_K = K_post - K_hat is
        # computed in the same post-RoPE coordinate system the CARE-KV
        # correction operates in. V is never rotated, so it uses the
        # normal per-token quant. K_hat_pre_override holds the rotated
        # K_hat per kv_head (None for every other path → unchanged).
        K_hat_pre_override = None
        V_hat_pre_override = None
        bq_name_pf = getattr(cfg, "base_quantizer", "uniform")
        k_store_mode = getattr(cfg, "k_store_mode", "post_rope")
        # Pre-RoPE base-quant for kvquant_style AND the rotation quantizers
        # (rotatekv_style / randrot_style): apply the channel rotation to
        # PRE-RoPE K, then re-apply RoPE. Full RoPE only (rot == D).
        _prerope_ok = (k_store_mode == "pre_rope" and rope_applied and rot == D)
        if _prerope_ok and bq_name_pf == "kvquant_style":
            k_bits_pf, v_bits_pf = self._kivi_bits()
            K_hat_pre = _kivi_quant_dequant_K(K, k_bits_pf)        # (Hkv, T, D)
            V_hat_pre_override = _kivi_quant_dequant_V(V, v_bits_pf)
        elif _prerope_ok and bq_name_pf in _ROTATION_BASE_QUANTIZERS:
            k_bits_pf, v_bits_pf = self._kivi_bits()
            K_hat_pre, V_hat_pre_override = _dispatch_base_kv_quant(
                bq_name_pf, K, V, k_bits_pf, v_bits_pf)           # (Hkv, T, D)
        else:
            K_hat_pre = None
        if _prerope_ok and K_hat_pre is not None:
            _, k_hat_b = apply_rotary_pos_emb(
                Q.unsqueeze(0), K_hat_pre.unsqueeze(0), cos, sin,
            )
            K_hat_pre_override = k_hat_b[0]                        # (Hkv, T, D) post-RoPE

        # Mixed-precision base override (LeanKV/MiKV-style, gated). Quantize post-RoPE
        # K/V with per-token bit-widths and feed it as the store's K̂/V̂ override so the
        # residual R = X − X̂_mixed AND the base attention both use the SAME mixed base
        # (consistent → CARE-KV stacks cleanly). Default off → (None,None) → unchanged.
        if rope_applied:
            _K_mp, _V_mp = self._mixed_precision_base(K_post, V)
            if _K_mp is not None:
                K_hat_pre_override, V_hat_pre_override = _K_mp, _V_mp

        # ── Cache write (always; post-RoPE K) ─────────────────────────
        # Returns the dequantized base K_hat/V_hat actually stored (valid rows).
        K_hat_kvh: List[Tensor] = []      # per kv_head, (T, D)
        V_hat_kvh: List[Tensor] = []
        for kv_h in range(Hkv):
            K_hat, V_hat = self._write_pages_for_kv_head(
                cache, kv_h, K_post[kv_h], V[kv_h],
                token_start_abs=token_start_abs,
                sink_tokens=sink_tokens,
                K_hat_seq_override=(None if K_hat_pre_override is None
                                    else K_hat_pre_override[kv_h]),
                V_hat_seq_override=(None if V_hat_pre_override is None
                                    else V_hat_pre_override[kv_h]),
            )
            K_hat_kvh.append(K_hat)
            V_hat_kvh.append(V_hat)

        K_hat_stack = torch.stack(K_hat_kvh, dim=0)     # (Hkv, T, D)
        V_hat_stack = torch.stack(V_hat_kvh, dim=0)     # (Hkv, T, D)

        # ── Phase-0 low-rank dense correction (eval-mode diagnostic) ──
        # CAREKV_LOWRANK_RANK=r>0 adds X̂' = X̂ + R P Pᵀ (top-r SVD subspace of
        # the residual). Upper-bound quality (exact SVD subspace + fp coeffs),
        # gated to base_quant so it is isolated from the sparse path.
        lr_rank = int(os.environ.get("CAREKV_LOWRANK_RANK", "0") or "0")
        if lr_rank > 0 and prefill_mode == "base_quant":
            K_hat_stack, V_hat_stack = self._lowrank_correct_eval(
                K_post, V, K_hat_stack, V_hat_stack, lr_rank,
                os.environ.get("CAREKV_LOWRANK_KIND", "both").lower())

        # ── Compute attention output for the chosen mode ──────────────
        if prefill_mode == "fp":
            # Full precision K_post / V — upper bound.
            K_attn = K_post.repeat_interleave(kv_group, dim=0) if Hq != Hkv else K_post   # (Hq, T, D)
            V_attn = V.repeat_interleave(kv_group, dim=0)      if Hq != Hkv else V        # (Hq, T, D)
            O_fp, _, _ = self._causal_attention_pure(Q_post, K_attn, V_attn, attention_mask)
            output = O_fp.transpose(0, 1).contiguous().reshape(T, Hq * D) @ self.W_O.T
            return output

        # base_quant / carekv_eval / carekv_stored all use K_hat / V_hat for base attention.
        K_attn = K_hat_stack.repeat_interleave(kv_group, dim=0) if Hq != Hkv else K_hat_stack
        V_attn = V_hat_stack.repeat_interleave(kv_group, dim=0) if Hq != Hkv else V_hat_stack
        O_base, attn_w, s_base = self._causal_attention_pure(
            Q_post, K_attn, V_attn, attention_mask,
        )
        # O_base: (Hq, T, D),  attn_w: (Hq, T, T),  s_base: (Hq, T, T)

        # Token eviction (SnapKV/H2O-style, gated). Applied to the BASE attention so
        # that base output, the residual router (attention-mass scoring), and the
        # sparse correction all operate on the evicted set — letting us test whether
        # CARE-KV's residual gain is ADDITIVE on top of eviction (Section 2 orthogonality).
        # Default CAREKV_EVICT_KEEP_RATIO unset/>=1.0 → no-op (byte-identical).
        O_base, attn_w, s_base = self._apply_token_eviction(O_base, attn_w, s_base, V_attn)

        if prefill_mode == "base_quant":
            output = O_base.transpose(0, 1).contiguous().reshape(T, Hq * D) @ self.W_O.T
            return output

        if prefill_mode == "carekv_eval":
            # Need r_k / r_v in memory — recompute them from post-RoPE K, V and the K_hat/V_hat.
            r_k = (K_post - K_hat_stack).repeat_interleave(kv_group, dim=0) if Hq != Hkv \
                else (K_post - K_hat_stack)
            r_v = (V - V_hat_stack).repeat_interleave(kv_group, dim=0) if Hq != Hkv \
                else (V - V_hat_stack)
            o_corr = self._apply_sparse_prefill_correction_eval(
                o_base=O_base.unsqueeze(0),
                attn_w=attn_w.unsqueeze(0),
                q_b=Q_post.unsqueeze(0),
                v_base=V_attn.unsqueeze(0),
                r_k=r_k.unsqueeze(0),
                r_v=r_v.unsqueeze(0),
            ).squeeze(0)
            output = o_corr.transpose(0, 1).contiguous().reshape(T, Hq * D) @ self.W_O.T
            return output

        if prefill_mode == "carekv_stored":
            # Decide whether any positive read budget is configured.  Absolute
            # mode bypasses the ratio short-circuit; in ratio mode either
            # global or per-kind > 0 keeps the correction active.
            any_read_budget = (
                (cfg.read_budget_mode in ("absolute", "adaptive_score")
                 and (cfg.read_abs_k > 0 or cfg.read_abs_v > 0))
                or cfg.read_budget_ratio > 0
                or (cfg.read_budget_ratio_k or 0) > 0
                or (cfg.read_budget_ratio_v or 0) > 0
            )
            if not any_read_budget:
                # Short-circuit: read budget=0 across the board must give
                # exactly base_quant (R=0 invariant preserved).
                output = O_base.transpose(0, 1).contiguous().reshape(T, Hq * D) @ self.W_O.T
                return output
            effective_stats = debug_stats
            if effective_stats is None and _debug_stats_enabled():
                effective_stats = _DEBUG_STATS
            o_corr = self._apply_sparse_prefill_correction_stored(
                cache=cache,
                o_base=O_base,
                attn_w=attn_w,
                q_post=Q_post,
                s_base=s_base,
                V_base_full=V_hat_stack,    # (Hkv, T, D)
                debug_stats=effective_stats,
            )
            output = o_corr.transpose(0, 1).contiguous().reshape(T, Hq * D) @ self.W_O.T
            return output

        raise ValueError(f"Unknown CAREKV_PREFILL_MODE={prefill_mode}")

    # ─────────────────────────────────────────
    # Causal attention (pure tensor; used for base attention computation)
    # ─────────────────────────────────────────

    @staticmethod
    def _mixed_precision_base(K_post, V):
        """LeanKV/MiKV-style mixed-precision base (EXPERIMENTAL, gated). Quantizes the
        post-RoPE K and V with PER-TOKEN bit-widths: the top CAREKV_MIXEDPREC_HI_FRAC of
        tokens (by saliency) at bits_hi, the rest at bits_lo. Returns (K_mp, V_mp) — the
        dequantized mixed-precision base — to be passed as the store's K̂/V̂ override so
        the residual R = X − X̂_mixed is computed against this base AND attention uses it
        (consistent). Default unset / hi_frac∉(0,1) → returns (None, None) → no-op
        (byte-identical; preserves READ=0≡base and Gate A/B).

        Env: CAREKV_MIXEDPREC_HI_FRAC, CAREKV_MIXEDPREC_BITS_HI/LO,
             CAREKV_MIXEDPREC_SALIENCY (vnorm | recent | random).
        """
        try:
            hi_frac = float(os.environ.get("CAREKV_MIXEDPREC_HI_FRAC", "0") or 0)
        except Exception:
            hi_frac = 0.0
        if not (0.0 < hi_frac < 1.0):
            return None, None
        bits_hi = int(os.environ.get("CAREKV_MIXEDPREC_BITS_HI", "4"))
        bits_lo = int(os.environ.get("CAREKV_MIXEDPREC_BITS_LO", "3"))
        sal = os.environ.get("CAREKV_MIXEDPREC_SALIENCY", "vnorm")
        Hkv, T, D = V.shape
        dev = V.device
        n_hi = max(1, min(T, int(round(hi_frac * T))))
        if sal == "recent":
            score = torch.arange(T, device=dev, dtype=torch.float32)
        elif sal == "random":
            score = torch.rand(T, device=dev)
        else:  # vnorm: per-token magnitude (mean over kv-heads of ||V[:,t,:]||)
            score = V.float().norm(dim=2).mean(dim=0)        # (T,)
        hi_mask = torch.zeros(T, dtype=torch.bool, device=dev)
        hi_mask[torch.topk(score, n_hi).indices] = True
        m = hi_mask.view(1, T, 1)
        K_hi = _kivi_quant_dequant_K(K_post, bits_hi); K_lo = _kivi_quant_dequant_K(K_post, bits_lo)
        V_hi = _kivi_quant_dequant_V(V, bits_hi);      V_lo = _kivi_quant_dequant_V(V, bits_lo)
        K_mp = torch.where(m, K_hi.to(K_post.dtype), K_lo.to(K_post.dtype))
        V_mp = torch.where(m, V_hi.to(V.dtype), V_lo.to(V.dtype))
        return K_mp, V_mp

    @staticmethod
    def _apply_token_eviction(O_base, attn_w, s_base, V_attn):
        """SnapKV/H2O-style token eviction (EXPERIMENTAL, gated). Keeps the top-K key
        tokens by accumulated attention (heavy hitters) + a recent window + sink
        tokens; masks the rest and re-normalizes. Recomputes (O_base, attn_w, s_base)
        so every downstream consumer (base output, residual router, sparse correction)
        sees the evicted attention. Default unset/keep_ratio>=1.0 → returns inputs
        unchanged (byte-identical; preserves the READ=0≡base and Gate A/B invariants).

        Env: CAREKV_EVICT_KEEP_RATIO (fraction of keys kept; <1 enables),
             CAREKV_EVICT_POLICY (h2o | recent_only), CAREKV_EVICT_RECENT (window),
             CAREKV_EVICT_SINK (first-N always kept).
        """
        try:
            r = float(os.environ.get("CAREKV_EVICT_KEEP_RATIO", "1.0"))
        except Exception:
            r = 1.0
        if not (0.0 < r < 1.0):
            return O_base, attn_w, s_base
        Hq, T, N = attn_w.shape
        dev = attn_w.device
        policy = os.environ.get("CAREKV_EVICT_POLICY", "h2o")
        recent = int(os.environ.get("CAREKV_EVICT_RECENT", str(max(1, N // 10))))
        sink = int(os.environ.get("CAREKV_EVICT_SINK", "4"))
        keep_n = max(1, int(round(r * N)))
        keep = torch.zeros(N, dtype=torch.bool, device=dev)
        keep[:min(sink, N)] = True
        if recent > 0:
            keep[max(0, N - recent):] = True
        if policy == "recent_only":
            score = torch.arange(N, device=dev, dtype=torch.float32)
        else:  # h2o / heavy_hitter: total attention received per key
            score = attn_w.float().sum(dim=(0, 1))             # (N,)
        rem = keep_n - int(keep.sum().item())
        if rem > 0:
            sc = score.clone(); sc[keep] = float("-inf")
            navail = int((~keep).sum().item())
            if navail > 0:
                top = torch.topk(sc, min(rem, navail)).indices
                keep[top] = True
        if bool(keep.all()):
            return O_base, attn_w, s_base
        evict = (~keep).view(1, 1, N)
        s_new = s_base.masked_fill(evict, float("-inf"))
        attn_new = torch.nan_to_num(torch.softmax(s_new, dim=-1), nan=0.0)
        O_new = torch.einsum("htn,hnd->htd", attn_new.to(V_attn.dtype), V_attn)
        return O_new, attn_new, s_new


    @staticmethod
    def _causal_attention_pure(
        Q: Tensor,                                  # (Hq, T, D)
        K: Tensor,                                  # (Hq, T, D)
        V: Tensor,                                  # (Hq, T, D)
        attention_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Returns O (Hq, T, D), attn_w (Hq, T, T), s (Hq, T, T pre-softmax).
        """
        Hq, T, D = Q.shape
        scores = torch.einsum("hid,hjd->hij", Q, K) / math.sqrt(D)
        if attention_mask is not None:
            am = attention_mask
            if am.dim() == 4:
                am = am.squeeze(0).squeeze(0)   # (T, T) expected
            elif am.dim() == 3:
                am = am.squeeze(0)
            scores = scores + am.to(device=scores.device, dtype=scores.dtype).unsqueeze(0)
        else:
            min_val = torch.finfo(scores.dtype).min
            causal = torch.full((T, T), min_val, device=scores.device, dtype=scores.dtype)
            causal = torch.triu(causal, diagonal=1)
            scores = scores + causal.unsqueeze(0)
        attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        out = torch.einsum("hij,hjd->hid", attn_w, V)
        return out, attn_w, scores

    # ─────────────────────────────────────────
    # Decode: one new token (use_cache=False legacy smoke path)
    # ─────────────────────────────────────────

    def decode_step(
        self,
        cache: CAREKVCache,
        hidden: Tensor,                                 # (1, model_dim)
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        debug_stats: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        One new token.  Appends K/V into the open page for each KV head
        (alloc a new page only when the current one is full).
        """
        cfg = self.cfg
        dev = hidden.device
        Hq = int(cfg.num_heads)
        Hkv = int(cfg.num_kv_heads)
        D = int(cfg.head_dim)
        kv_group = Hq // Hkv

        hidden_1 = hidden.squeeze(0)
        q = (hidden_1 @ self.W_Q.T).reshape(Hq, D)
        k = (hidden_1 @ self.W_K.T).reshape(Hkv, D)
        v = (hidden_1 @ self.W_V.T).reshape(Hkv, D)

        # Position is the next absolute token index for kv_head 0.
        pos_abs = cache.total_valid_tokens(self.layer_id, 0)

        cos = sin = None
        rope_applied_decode = (position_embeddings is not None
                               and apply_rotary_pos_emb is not None)
        if rope_applied_decode:
            cos, sin = position_embeddings
            cos = cos.to(device=dev, dtype=q.dtype)
            sin = sin.to(device=dev, dtype=q.dtype)
            q_b = q.unsqueeze(0).unsqueeze(2)    # (1, Hq, 1, D)
            k_b = k.unsqueeze(0).unsqueeze(2)    # (1, Hkv, 1, D)
            q_b, k_b = apply_rotary_pos_emb(q_b, k_b, cos, sin)
            q_post = q_b.squeeze(0).squeeze(1)   # (Hq, D)
            k_post = k_b.squeeze(0).squeeze(1)   # (Hkv, D)
        else:
            q_post = q
            k_post = k

        use_kivi_decode = getattr(cfg, "base_quantizer", "uniform") in _SIDE_BUFFER_BASE_QUANTIZERS
        # Pre-RoPE decode (kvquant_style + rotation quantizers): quantize the
        # pre-RoPE k row, then re-apply RoPE to match the stored post-RoPE layout.
        prerope_decode = (getattr(cfg, "k_store_mode", "post_rope") == "pre_rope"
                          and rope_applied_decode
                          and getattr(cfg, "base_quantizer", "uniform")
                              in ({"kvquant_style"} | _ROTATION_BASE_QUANTIZERS))
        # Append per KV head into open page (alloc when needed).
        for kv_h in range(Hkv):
            page_id = cache.get_open_page(self.layer_id, kv_h)
            if page_id is None:
                # Alloc a new page; the first token's codes get written into row 0.
                page_id = cache.alloc_page(self.layer_id, kv_h)
                # Initialize an empty buffer in the page (zeros are already there
                # from buffer init; valid_tokens starts at 0).
                cache.valid_tokens[self.layer_id, kv_h, page_id] = 0

            if use_kivi_decode:
                # Decode-mode base-quant of a single token: per-channel K
                # over T=1 reduces to scale=|K|/qmax → near-zero quant
                # error. This is a documented prototype limitation
                # (real KIVI / RotateKV sees the whole sequence at
                # prefill time). Dispatch on cfg.base_quantizer.
                bq_name_decode = getattr(cfg, "base_quantizer", "uniform")
                k_bits, v_bits = self._kivi_bits()
                if prerope_decode:
                    # Quantize the PRE-RoPE k row, then re-apply RoPE. K uses the
                    # per-quantizer K function (kivi per-channel, or rotation
                    # rotate+quant); V is coordinate-independent.
                    if bq_name_decode in _ROTATION_BASE_QUANTIZERS:
                        K_hat_pre_row, V_hat_row = _dispatch_base_kv_quant(
                            bq_name_decode, k[kv_h:kv_h+1], v[kv_h:kv_h+1],
                            k_bits, v_bits)
                        V_hat_row = V_hat_row.to(v.dtype)      # (1, D)
                    else:  # kvquant_style (per-channel K)
                        K_hat_pre_row = _kivi_quant_dequant_K(
                            k[kv_h:kv_h+1], k_bits)            # (1, D) pre-RoPE
                        V_hat_row = _kivi_quant_dequant_V(
                            v[kv_h:kv_h+1], v_bits).to(v.dtype)
                    _, k_hat_b = apply_rotary_pos_emb(
                        q[kv_h:kv_h+1].unsqueeze(0).unsqueeze(2),
                        K_hat_pre_row.unsqueeze(0).unsqueeze(2), cos, sin,
                    )
                    K_hat_row = k_hat_b.squeeze(0).squeeze(1).to(k_post.dtype)  # (1, D)
                else:
                    K_hat_seq_dec, V_hat_seq_dec = _dispatch_base_kv_quant(
                        bq_name_decode,
                        k_post[kv_h:kv_h+1], v[kv_h:kv_h+1],
                        k_bits, v_bits,
                    )
                    K_hat_row = K_hat_seq_dec.squeeze(0).to(k_post.dtype)
                    V_hat_row = V_hat_seq_dec.squeeze(0).to(v.dtype)
                offset = cache.append_to_page_kivi(
                    self.layer_id, kv_h, page_id,
                    K_hat_row, V_hat_row,
                )
            else:
                # Single-row codes (uniform path, unchanged).
                k_codes_full, k_scale_full, _ = self._quantize_dequantize(k_post[kv_h:kv_h+1])
                v_codes_full, v_scale_full, _ = self._quantize_dequantize(v[kv_h:kv_h+1])
                # quantize/dequantize returns (1, D) and (1, G).
                offset = cache.append_to_page(
                    self.layer_id, kv_h, page_id,
                    k_codes_full[0], k_scale_full[0],
                    v_codes_full[0], v_scale_full[0],
                )

            # If this append just filled the page, process residuals for the page.
            if int(cache.valid_tokens[self.layer_id, kv_h, page_id].item()) == cfg.page_size:
                if use_kivi_decode:
                    K_hat_pg = cache.base_K_hat_fp16[self.layer_id, kv_h, page_id].to(k_post.dtype)
                    V_hat_pg = cache.base_V_hat_fp16[self.layer_id, kv_h, page_id].to(v.dtype)
                    self.store_manager.process_page(
                        cache=cache, kv_head=kv_h, page_id=page_id,
                        K_orig=K_hat_pg, V_orig=V_hat_pg,
                        K_codes=None, K_scale=None,
                        V_codes=None, V_scale=None,
                        token_start=pos_abs + 1 - cfg.page_size,
                        num_valid=cfg.page_size,
                        is_recent=True, is_sink=(pos_abs + 1 - cfg.page_size) < 4,
                        K_hat_override=K_hat_pg, V_hat_override=V_hat_pg,
                    )
                    continue
                # Reconstruct page tensors from the cache to call process_page.
                # Honor packed_base — base_K/V_codes hold packed bytes in that
                # mode, which must be unpacked before dequantize() can read
                # them as (T, D) int8 codes.  scale_quant=int8 likewise needs
                # decoded scales for downstream consumers.
                K_codes_pg = cache._maybe_unpack_rows(
                    cache.base_K_codes[self.layer_id, kv_h, page_id]
                )                                                            # (T, D) int8
                V_codes_pg = cache._maybe_unpack_rows(
                    cache.base_V_codes[self.layer_id, kv_h, page_id]
                )
                if cfg.scale_quant == "int8":
                    K_scale_pg = cache._maybe_dequant_scale(
                        cache.base_K_scale[self.layer_id, kv_h, page_id],
                        cache.base_K_scale_master[self.layer_id, kv_h, page_id],
                        torch.float16,
                    )
                    V_scale_pg = cache._maybe_dequant_scale(
                        cache.base_V_scale[self.layer_id, kv_h, page_id],
                        cache.base_V_scale_master[self.layer_id, kv_h, page_id],
                        torch.float16,
                    )
                else:
                    K_scale_pg = cache.base_K_scale[self.layer_id, kv_h, page_id].to(torch.float16)
                    V_scale_pg = cache.base_V_scale[self.layer_id, kv_h, page_id].to(torch.float16)
                K_hat_pg = dequantize(
                    K_codes_pg, K_scale_pg, (cfg.page_size, D), self.base_qcfg,
                ).to(k_post.dtype)
                V_hat_pg = dequantize(
                    V_codes_pg, V_scale_pg, (cfg.page_size, D), self.base_qcfg,
                ).to(v.dtype)
                # We need K_orig / V_orig for residual computation.  We don't
                # have them streaming; reuse K_hat as an upper-bound (residual
                # then resolves to a near-zero block, so this page contributes
                # negligibly to corrections).  Acceptable for decode-mode usage.
                self.store_manager.process_page(
                    cache=cache, kv_head=kv_h, page_id=page_id,
                    K_orig=K_hat_pg, V_orig=V_hat_pg,
                    K_codes=K_codes_pg, K_scale=K_scale_pg,
                    V_codes=V_codes_pg, V_scale=V_scale_pg,
                    token_start=pos_abs + 1 - cfg.page_size,
                    num_valid=cfg.page_size,
                    is_recent=True, is_sink=(pos_abs + 1 - cfg.page_size) < 4,
                )

        # Run attention for this query.
        kind = os.environ.get("CAREKV_PREFILL_RESIDUAL_KIND", "both").lower()
        k_corr_scale = float(os.environ.get("CAREKV_K_CORRECTION_SCALE", "0.1"))
        score_normalize = os.environ.get("CAREKV_SCORE_NORMALIZE", "0") == "1"
        effective_stats = debug_stats
        if effective_stats is None and _debug_stats_enabled():
            effective_stats = _DEBUG_STATS
        O = self.attn.forward(
            cache, q_post,
            kind=kind, k_corr_scale=k_corr_scale,
            score_normalize=score_normalize,
            debug_stats=effective_stats,
        )
        O_flat = O.reshape(1, Hq * D)
        output = O_flat @ self.W_O.T
        return output
