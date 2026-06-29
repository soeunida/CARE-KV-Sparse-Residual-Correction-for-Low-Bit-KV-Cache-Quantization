"""
care_kv/residual_store.py
--------------------------
Store-time residual filtering.

When a new page of KV tokens is written into the cache:
  1. Compute R_K = K - dequant(K_base),  R_V = V - dequant(V_base)
     (only the leading num_valid rows of each are real tokens)
  2. Enumerate residual candidates (K: per channel_group, V: per token_block)
  3. Score each candidate with prior_score = error_norm × sensitivity × structural_prior
  4. Keep only top-N candidates within store_budget
  5. Pack selected residuals into fixed-size slots and write to cache

Residual packing: 4-bit symmetric, stored as int8 with pairs of values packed.
"""

from __future__ import annotations
import os
import torch
from torch import Tensor
from typing import List, Tuple, Optional
import math

from .cache import CAREKVCache, CacheConfig, PageMeta, layer_budget_multiplier
from .quantizer import QuantConfig, quantize, dequantize


# ─────────────────────────────────────────────
# Env-driven store-time policy
#
# CAREKV_STORE_POLICY:
#   joint                — single shared budget across (K ∪ V); legacy.
#   per_kind             — always split into separate K and V budgets,
#                          each `max(1, int(total_of_that_kind * ratio))`.
#   residual_kind_aware  — honor CAREKV_PREFILL_RESIDUAL_KIND:
#                          "v"   → store V only
#                          "k"   → store K only
#                          "both"→ per_kind split
#                          (default — paper-ready)
#
# CAREKV_STORE_MIN_PER_KIND (default 1): floor on per-kind budget when ratio > 0.
# CAREKV_STORE_K_FRACTION, CAREKV_STORE_V_FRACTION: optional per-kind override
#   multipliers applied to store_budget_ratio (default 1.0 each).  Useful for
#   biasing storage toward one residual type without changing the overall
#   budget knob.
# ─────────────────────────────────────────────

import random as _random


def _position_weight(policy: str, tok_pos: int, seq_len: int) -> float:
    """Phase 11B position-aware weight for a V token block (token_pos in [0,seq_len)).
    Preferred positions → weight 1.0; de-emphasized → 0.05 (still selectable if budget
    remains, so total budget stays comparable). K candidates are not token-positioned."""
    if seq_len <= 0:
        return 1.0
    frac = tok_pos / seq_len
    prefix = min(64, max(1, int(0.10 * seq_len)))
    recent = 0.75
    HI, LO = 1.0, 0.05
    if policy == "recent_token":
        return HI if frac >= recent else LO
    if policy == "prefix_sink":
        return HI if tok_pos < prefix else LO
    if policy == "sink_plus_recent":
        return HI if (tok_pos < prefix or frac >= recent) else LO
    if policy == "middle_drop":
        # de-emphasize only the central band; keep front + back generously
        return LO if (0.375 <= frac < 0.625) else HI
    return 1.0


def _phase11b_rescore(k_cands, v_cands, k_errs, v_errs, layer_id, page_id, token_start, cfg):
    """Gated selector-variant / position-policy candidate re-scoring (EXPERIMENTAL).
    variant in {current, random, oracle_residual_magnitude, oracle_reconstruction_error};
    position in {none, recent_token, prefix_sink, sink_plus_recent, middle_drop}.
    NO-OP when both are default → default behavior byte-identical."""
    variant = os.environ.get("CAREKV_SELECTOR_VARIANT", "current")
    pos = os.environ.get("CAREKV_POSITION_POLICY", "none")
    if variant == "current" and pos == "none":
        return
    seq_len = int(os.environ.get("CAREKV_SEQ_LEN", "0") or 0)
    for cands, errs, kind in ((k_cands, k_errs, "K"), (v_cands, v_errs, "V")):
        for i, c in enumerate(cands):
            en = float(errs[i]) if i < len(errs) else 0.0
            if en <= 0:
                continue                                   # keep zero-error slots at 0
            if variant == "random":
                rng = _random.Random(layer_id * 100003 + page_id * 131 + i * 17 + (0 if kind == "K" else 7))
                base = rng.random()
            elif variant == "oracle_residual_magnitude":
                base = en                                  # rank by raw ||R||
            elif variant == "oracle_reconstruction_error":
                base = en * en                             # emphasize largest reconstruction errors
            else:                                          # "current"
                base = c.prior_score
            w = 1.0
            if pos != "none" and kind == "V":              # position is a token concept (V blocks)
                tok = token_start + c.group_idx * cfg.v_token_block
                w = _position_weight(pos, tok, seq_len)
            c.prior_score = base * w


def _resolve_store_policy() -> Tuple[str, str, int, float, float]:
    policy = os.environ.get("CAREKV_STORE_POLICY", "residual_kind_aware").lower()
    kind   = os.environ.get("CAREKV_PREFILL_RESIDUAL_KIND", "both").lower()
    mpk    = int(os.environ.get("CAREKV_STORE_MIN_PER_KIND", "1"))
    kf     = float(os.environ.get("CAREKV_STORE_K_FRACTION", "1.0"))
    vf     = float(os.environ.get("CAREKV_STORE_V_FRACTION", "1.0"))
    if policy not in {"joint", "per_kind", "residual_kind_aware"}:
        raise ValueError(f"Unknown CAREKV_STORE_POLICY={policy}")
    if kind not in {"v", "k", "both"}:
        raise ValueError(f"Unknown CAREKV_PREFILL_RESIDUAL_KIND={kind}")
    return policy, kind, mpk, kf, vf


# ─────────────────────────────────────────────
# 4-bit packing helpers
# ─────────────────────────────────────────────

def pack_4bit(x: Tensor, scale_hint: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    Quantize x to 4-bit symmetric and pack pairs into int8.
    x: arbitrary shape float
    Returns: packed int8 (ceil(numel/2),), scale (fp16, shape (1,))
    """
    max_abs = x.abs().max().clamp(min=1e-8)
    s = max_abs / 7.0
    codes = (x / s).round().clamp(-8, 7).to(torch.int8)
    flat = codes.flatten()
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, flat.new_zeros(1)])
    lo = flat[0::2] & 0x0F
    hi = (flat[1::2] << 4) & 0xF0
    packed = (lo | hi).to(torch.int8)
    return packed, s.to(torch.float16).unsqueeze(0)


def unpack_4bit(packed: Tensor, scale: Tensor, numel: int) -> Tensor:
    """Unpack int8 → 4-bit signed values → float."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    interleaved = torch.stack([lo, hi], dim=-1).flatten()[:numel]
    signed = interleaved.to(torch.int8)
    signed[signed > 7] -= 16
    return signed.float() * scale.float()


# ─────────────────────────────────────────────
# Residual candidate record
# ─────────────────────────────────────────────

class ResidualCandidate:
    __slots__ = ["kind", "page_id", "kv_head", "group_idx",
                 "payload", "scale", "sketch", "prior_score", "numel"]

    def __init__(self, kind, page_id, kv_head, group_idx,
                 payload, scale, sketch, prior_score, numel):
        self.kind = kind            # "K" or "V"
        self.page_id = page_id
        self.kv_head = kv_head
        self.group_idx = group_idx
        self.payload = payload
        self.scale = scale
        self.sketch = sketch
        self.prior_score = prior_score
        self.numel = numel


# ─────────────────────────────────────────────
# Sketch projection matrix (fixed random, per head_dim)
# ─────────────────────────────────────────────

_sketch_proj_cache: dict = {}

def get_sketch_proj(head_dim: int, k_channel_group: int,
                    sketch_dim: int, device: torch.device) -> Tensor:
    """Returns a fixed random projection matrix (k_channel_group, sketch_dim)."""
    key = (head_dim, k_channel_group, sketch_dim, str(device))
    if key not in _sketch_proj_cache:
        gen = torch.Generator(device="cpu").manual_seed(42)
        P_cpu = torch.randn(k_channel_group, sketch_dim, generator=gen) / math.sqrt(sketch_dim)
        _sketch_proj_cache[key] = P_cpu.to(device)
    return _sketch_proj_cache[key]


# ─────────────────────────────────────────────
# Main store manager
# ─────────────────────────────────────────────

class ResidualStoreManager:
    """
    Manages store-time residual filtering for a single layer.
    Called once per new page of tokens (per KV head).
    """

    def __init__(self, cfg: CacheConfig, layer_id: int, device: torch.device):
        self.cfg = cfg
        self.layer_id = layer_id
        self.device = device
        self.sensitivity = cfg.layer_sensitivity[layer_id]

        self.quant_cfg_residual = QuantConfig(
            bits=cfg.residual_bits,
            group_size=cfg.group_size,
            symmetric=True,
        )

    def process_page(
        self,
        cache: CAREKVCache,
        kv_head: int,
        page_id: int,
        K_orig: Tensor,    # (T, D) float, original K for this page (post-RoPE, padded)
        V_orig: Tensor,    # (T, D) float
        K_codes: Tensor,   # (T, D) int8 base codes — None when K_hat_override is provided
        K_scale: Tensor,   # (T, G) fp16 base scale — None when K_hat_override is provided
        V_codes: Tensor,   # (T, D) int8 base codes — None when V_hat_override is provided
        V_scale: Tensor,   # (T, G) fp16 base scale — None when V_hat_override is provided
        token_start: int,
        num_valid: int,
        is_recent: bool = True,
        is_sink: bool = False,
        K_hat_override: Optional[Tensor] = None,   # (T, D) — bypass uniform dequant
        V_hat_override: Optional[Tensor] = None,   # (T, D)
    ) -> PageMeta:
        """
        Score residual candidates for this page and store the top fraction
        (store_budget_ratio).  Only the leading num_valid rows of K_orig /
        V_orig are treated as real tokens.

        When K_hat_override / V_hat_override are provided (Phase Q-stacked:
        base_quantizer='kivi_style'), residuals are computed against the
        provided K_hat / V_hat instead of dequantizing K_codes / V_codes.
        K_codes / K_scale may be None in that path.
        """
        cfg = self.cfg
        T, D = K_orig.shape
        assert 0 <= num_valid <= T, (num_valid, T)

        # ── Compute residuals ─────────────────────────────────────────
        if K_hat_override is not None and V_hat_override is not None:
            K_hat = K_hat_override.to(K_orig.dtype)
            V_hat = V_hat_override.to(V_orig.dtype)
        else:
            qcfg_base = QuantConfig(bits=cfg.base_bits, group_size=cfg.group_size)
            K_hat = dequantize(K_codes, K_scale, (T, D), qcfg_base)
            V_hat = dequantize(V_codes, V_scale, (T, D), qcfg_base)
        R_K = (K_orig - K_hat).float()   # (T, D)
        R_V = (V_orig - V_hat).float()   # (T, D)

        # Zero out padded rows so they cannot leak into norms/sketch.
        if num_valid < T:
            R_K[num_valid:] = 0
            R_V[num_valid:] = 0

        # ── Structural prior ──────────────────────────────────────────
        struct_prior = 1.0
        if is_recent:
            struct_prior *= 1.5
        if is_sink:
            struct_prior *= 2.0

        # ── Enumerate K residual candidates (per channel group) ──────
        num_cg = D // cfg.k_channel_group
        k_candidates: List[ResidualCandidate] = []
        k_error_norms = []

        for cg in range(num_cg):
            c_start = cg * cfg.k_channel_group
            c_end   = c_start + cfg.k_channel_group
            if num_valid > 0:
                rk_cg_valid = R_K[:num_valid, c_start:c_end]
                err_norm = rk_cg_valid.norm(dim=-1).mean().item()
                rk_mean = rk_cg_valid.mean(dim=0)
            else:
                err_norm = 0.0
                rk_mean = torch.zeros(cfg.k_channel_group, device=R_K.device)
            k_error_norms.append(err_norm)

            P = get_sketch_proj(D, cfg.k_channel_group, cfg.sketch_dim, R_K.device)
            sketch = (rk_mean @ P).to(torch.float16)

            prior = err_norm * self.sensitivity * struct_prior

            # Pack the full padded page so slot size is fixed; padded rows are 0.
            packed, scale = pack_4bit(R_K[:, c_start:c_end])

            k_candidates.append(ResidualCandidate(
                kind="K", page_id=page_id, kv_head=kv_head,
                group_idx=cg, payload=packed, scale=scale,
                sketch=sketch, prior_score=prior,
                numel=cfg.page_size * cfg.k_channel_group,
            ))

        # ── Enumerate V residual candidates (per token block) ────────
        num_vb = math.ceil(cfg.page_size / cfg.v_token_block)
        v_candidates: List[ResidualCandidate] = []
        v_error_norms = []

        for vb in range(num_vb):
            t_start = vb * cfg.v_token_block
            t_end_padded = t_start + cfg.v_token_block
            t_end_valid  = min(t_end_padded, num_valid)
            if t_end_valid > t_start:
                rv_blk_valid = R_V[t_start:t_end_valid]
                err_norm = rv_blk_valid.norm(dim=-1).mean().item()
            else:
                err_norm = 0.0
            v_error_norms.append(err_norm)

            prior = err_norm * self.sensitivity * struct_prior

            # Pack at the fixed slot size; pad with zeros if block crosses
            # the valid/padded boundary.
            rv_blk_padded = R_V[t_start:t_end_padded]
            if rv_blk_padded.shape[0] < cfg.v_token_block:
                pad = torch.zeros(
                    cfg.v_token_block - rv_blk_padded.shape[0],
                    D, device=R_V.device, dtype=R_V.dtype,
                )
                rv_blk_padded = torch.cat([rv_blk_padded, pad], dim=0)
            packed, scale = pack_4bit(rv_blk_padded)

            v_candidates.append(ResidualCandidate(
                kind="V", page_id=page_id, kv_head=kv_head,
                group_idx=vb, payload=packed, scale=scale,
                sketch=None, prior_score=prior,
                numel=cfg.v_token_block * D,
            ))

        # ── Apply per-page store budget under chosen policy ───────────
        # store_budget_ratio == 0 → store nothing (preserved invariant).
        # Otherwise route by CAREKV_STORE_POLICY:
        #   joint              — legacy shared budget over (K ∪ V)
        #   per_kind           — always split into K & V budgets
        #   residual_kind_aware— skip a kind entirely if it won't be read
        #                        (CAREKV_PREFILL_RESIDUAL_KIND=v/k); for "both"
        #                        falls back to per_kind split.
        store_policy, kind, min_per_kind, k_frac, v_frac = _resolve_store_policy()

        _phase11b_rescore(k_candidates, v_candidates, k_error_norms, v_error_norms,
                          self.layer_id, page_id, token_start, cfg)
        k_scored = [c for c in k_candidates if c.prior_score > 0]
        v_scored = [c for c in v_candidates if c.prior_score > 0]
        k_scored.sort(key=lambda c: c.prior_score, reverse=True)
        v_scored.sort(key=lambda c: c.prior_score, reverse=True)

        # Resolve per-kind store budgets honoring (a) absolute vs ratio
        # mode, (b) per-kind ratio overrides, (c) the residual_kind_aware
        # skip-a-kind logic, and (d) the joint shared-budget legacy path.
        use_k = True; use_v = True
        if store_policy == "residual_kind_aware":
            if kind == "v": use_k = False
            if kind == "k": use_v = False

        # Per-layer budget multiplier (Phase E).  Multiplies the resolved
        # per-kind budget; round() instead of int() so half-budgets bias up
        # rather than collapse to 0 for mid layers under u_shaped.
        lm = layer_budget_multiplier(cfg, self.layer_id)

        if cfg.store_budget_mode == "absolute":
            base_k = int(round(max(0, int(cfg.store_abs_k)) * lm))
            base_v = int(round(max(0, int(cfg.store_abs_v)) * lm))
            k_budget = max(0, base_k) if use_k else 0
            v_budget = max(0, base_v) if use_v else 0
            joint_budget = k_budget + v_budget
        else:
            # Ratio mode.  per-kind override wins; else fall back to global.
            def _res(o, fb): return fb if (o is None or o < 0) else o
            rk = _res(cfg.store_budget_ratio_k, cfg.store_budget_ratio) * k_frac * lm
            rv = _res(cfg.store_budget_ratio_v, cfg.store_budget_ratio) * v_frac * lm
            k_total = max(1, len(k_candidates))
            v_total = max(1, len(v_candidates))
            k_budget = (max(min_per_kind, int(k_total * rk))
                        if (use_k and rk > 0) else 0)
            v_budget = (max(min_per_kind, int(v_total * rv))
                        if (use_v and rv > 0) else 0)
            joint_budget = max(1, int((k_total + v_total) * cfg.store_budget_ratio * lm))

        if (cfg.store_budget_ratio <= 0
                and cfg.store_budget_mode != "absolute"
                and (cfg.store_budget_ratio_k or 0) <= 0
                and (cfg.store_budget_ratio_v or 0) <= 0):
            selected = []
        elif store_policy == "joint":
            joined = sorted(k_scored + v_scored, key=lambda c: c.prior_score, reverse=True)
            selected = joined[:joint_budget]
        else:
            selected = k_scored[:k_budget] + v_scored[:v_budget]
        k_budget_used = sum(1 for c in selected if c.kind == "K")
        v_budget_used = sum(1 for c in selected if c.kind == "V")

        # ── Write selected residuals to cache ─────────────────────────
        k_slots = [-1] * num_cg
        v_slots = [-1] * num_vb
        k_sketch_list: List[Optional[Tensor]] = [None] * num_cg

        for cand in selected:
            if cand.kind == "K":
                slot = cache.alloc_k_slot()
                cache.write_k_residual(slot, cand.payload, cand.scale)
                k_slots[cand.group_idx] = slot
                k_sketch_list[cand.group_idx] = cand.sketch
            else:
                slot = cache.alloc_v_slot()
                cache.write_v_residual(slot, cand.payload, cand.scale)
                v_slots[cand.group_idx] = slot

        # Record sketches for all CGs (sketches are tiny and useful for
        # ranking even when a payload was not stored).
        for cg, sk in enumerate(k_sketch_list):
            if sk is None:
                k_sketch_list[cg] = k_candidates[cg].sketch

        k_err_tensor = torch.tensor(k_error_norms, dtype=torch.float16, device=R_K.device)
        v_err_tensor = torch.tensor(v_error_norms, dtype=torch.float16, device=R_V.device)
        k_sketch_tensor = torch.stack(k_sketch_list, dim=0)

        meta = PageMeta(
            page_id=page_id,
            layer_id=self.layer_id,
            kv_head=kv_head,
            token_start=token_start,
            num_tokens=num_valid,
            k_residual_slots=k_slots,
            v_residual_slots=v_slots,
            k_error_norm=k_err_tensor,
            v_error_norm=v_err_tensor,
            k_sketch=k_sketch_tensor,
            prior_score=float(sum(c.prior_score for c in selected)),
        )
        cache.meta_table[self.layer_id][kv_head][page_id] = meta
        return meta
