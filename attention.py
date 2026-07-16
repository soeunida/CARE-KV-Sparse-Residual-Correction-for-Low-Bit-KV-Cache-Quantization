"""
care_kv/attention.py
---------------------
CARE-KV attention computation.

Two public entry points:

apply_slot_corrections(...)
    Module-level helper used by both decode-time attention and the
    carekv_stored prefill path.  Given base attention (O_base, a_base, s_base,
    V_base) plus a query q, route over stored residual slots and accumulate
    ΔO_V and ΔO_K — using only stored slot reads, no full R_K/R_V tensors.

CAREKVAttention
    Single query-head decode step.  Reads base K/V from cache, computes base
    attention, then calls apply_slot_corrections.

CAREKVMultiHeadAttention
    Iterates query heads, mapping each to the appropriate KV head.

Correction formulas:
    ΔO_V = Σ_t a_t · R_V,t                              (sum over selected token blocks)
    ΔO_K ≈ Σ_t a_t · (q · R_K,t) · (V_base,t − O_base)   (1st-order Jacobian)

`CAREKV_K_CORRECTION_MODE=exact` replaces the Jacobian with the exact softmax
renormalization — see exact_softmax_correction below.
"""

from __future__ import annotations
import os
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict, Any
import math

from .cache import CAREKVCache, CacheConfig, PageMeta
from .quantizer import QuantConfig, dequantize
from .residual_router import ResidualRouter, _resolve_read_budgets, EPS
from .residual_store import unpack_4bit, unpack_residual


# ─────────────────────────────────────────────────────────────────────────
# Adaptive Output-Aware CARE-KV — gated helpers (sections B & C).
# OFF by default (raw-error scoring + uniform budget remain the defaults), so
# the READ=0 invariant and Gate A/B are unaffected unless CAREKV_RESIDUAL_SCORE
# =attention_output / CAREKV_BUDGET_POLICY=layer_head_impact are set explicitly.
# Both are pure functions, unit-tested on synthetic tensors.
# ─────────────────────────────────────────────────────────────────────────

# Diagnostic recorder for chunked-vs-full equivalence debugging. When set to a
# list (via debug_chunked_carekv_equivalence), every vectorized_joint_correction
# call appends its (layer, kv_head, q_offset, delta, scores, selections). None in
# normal operation (zero overhead).
_CHUNK_REC = None


def set_chunk_recorder(rec):
    """Set/clear the chunked-equivalence diagnostic recorder (list or None)."""
    global _CHUNK_REC
    _CHUNK_REC = rec


# ─────────────────────────────────────────────────────────────────────────
# Exact K correction (CAREKV_K_CORRECTION_MODE=exact)
#
# The default ΔO_K is the 1st-order Jacobian of softmax w.r.t. the key
# residual.  Writing δs_t = (q · R_K,t)/√D for the exact logit perturbation
# carried by the *selected* K slots, that Jacobian is the δs→0 limit of
#
#     a_new = softmax(s_base + δs) = a_base·e^δs / Σ_u a_base,u·e^δs,u
#     O_new = Σ_t a_new,t · (V_base,t + R_V,t·[t selected])
#
# which is directly computable from the same slot reads: no full R_K/R_V and
# no extra reads, one exp + one matmul over (Q, N).  The linear form diverges
# once |δs| ≳ 1 — exactly the outlier-heavy-K regime where CARE-KV loses to
# rotation-based baselines — because it extrapolates a convex exponential off
# its tangent.  a_new is a softmax, so the exact form is bounded by
# construction and needs no `k_corr_scale` damping (a global 0.1 that happens
# to stand in for the 1/√D the linear apply path omits).
#
# Two properties the callers rely on:
#   - no slots selected ⇒ δs ≡ 0, R_V ≡ 0 ⇒ a_new == a_base ⇒ ΔO == 0 exactly.
#     This preserves the READ=0 ≡ base_quant invariant bit-for-bit.
#   - kind=="v" ⇒ δs ≡ 0 ⇒ ΔO == Σ_t a_base,t·R_V,t, i.e. exact mode is a
#     no-op relabelling of the existing V-only correction.
# ─────────────────────────────────────────────────────────────────────────

def k_correction_mode() -> str:
    m = os.environ.get("CAREKV_K_CORRECTION_MODE", "linear").lower()
    if m not in {"linear", "exact"}:
        raise ValueError(f"Unknown CAREKV_K_CORRECTION_MODE={m}")
    return m


def exact_softmax_correction(
    A: Tensor,                      # (Q, N) base attention weights, rows sum to 1
    ds: Tensor,                     # (Q, N) exact logit perturbation from K slots
    V_base: Tensor,                 # (N, D) dequantized base V
    O_base: Tensor,                 # (Q, D) base attention output
    RV_full: Optional[Tensor] = None,   # (N, D) per-token V residual (selected rows)
    sel_tok: Optional[Tensor] = None,   # (Q, N) bool, which tokens carry a read V slot
) -> Tensor:
    """ΔO (Q, D) from renormalizing the base softmax under the exact δs.

    Computes in float32 regardless of caller dtype — the cached decode path
    passes fp16 V_base/O_base, while the vectorized prefill path pre-casts to
    float; both must work. Returns ΔO in O_base's original dtype.
    """
    out_dtype = O_base.dtype
    A = A.float(); ds = ds.float(); V_base = V_base.float(); O_base = O_base.float()
    w = torch.exp(ds - ds.amax(dim=1, keepdim=True))
    a_new = A * w
    a_new = a_new / a_new.sum(dim=1, keepdim=True).clamp(min=1e-30)
    O_new = a_new @ V_base
    if RV_full is not None and sel_tok is not None:
        O_new = O_new + (a_new * sel_tok.to(a_new.dtype)) @ RV_full.float()
    return (O_new - O_base).to(out_dtype)


def attention_output_residual_scores(A: Tensor, resid: Tensor, kind: str) -> Tensor:
    """Attention-output-aware residual impact score (spec section B).

    kind == "v":  A is (Q, T) base (quant) attention over T key positions;
                  `resid` is (T, N, D) per-slot V residual blocks (V_fp16 −
                  V_quant). Returns (N,) = ‖A @ resid_slot‖_2 per slot.
    kind == "k":  A is (Q, T) the attention DIFFERENCE (A_fp16 − A_quant);
                  `resid` is (T, D) = V_quant. Returns (Q,) = ‖ΔA @ V_quant‖_2.

    Scores are finite and non-negative (vector norms, non-finite zeroed). Used
    only to rank residuals — never changes correction values or the READ=0 path.
    """
    A = torch.nan_to_num(A.float(), nan=0.0, posinf=0.0, neginf=0.0)
    resid = torch.nan_to_num(resid.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if kind == "v":
        out = torch.einsum("qt,tnd->qnd", A, resid)                 # (Q, N, D)
        return out.permute(1, 0, 2).reshape(out.shape[1], -1).norm(dim=1).clamp_min(0.0)  # (N,)
    elif kind == "k":
        return (A @ resid).norm(dim=1).clamp_min(0.0)               # (Q,)
    else:
        raise ValueError(f"unknown kind {kind!r}")


def allocate_impact_budget(impact: Tensor, total: int) -> Tensor:
    """Distribute integer `total` across units proportional to non-negative
    `impact` (spec section C). Largest-remainder (Hamilton) apportionment:
    floor the proportional shares, hand leftover to the largest fractional
    remainders. Guarantees sum(alloc)==total, alloc>=0, near-zero units may get
    0, and the dominant-impact unit receives the maximum share.
    """
    impact = torch.nan_to_num(impact.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    n = impact.numel(); total = int(total)
    if total <= 0 or n == 0:
        return torch.zeros(n, dtype=torch.long)
    s = impact.sum()
    shares = (impact / s * total) if s > 0 else torch.full((n,), total / n)
    floor = torch.floor(shares).to(torch.long)
    rem = total - int(floor.sum().item())
    if rem > 0:
        frac = shares - floor.to(shares.dtype)
        order = torch.argsort(frac + impact * 1e-9, descending=True)  # ties → larger impact
        for i in range(rem):
            floor[order[i % n]] += 1
    return floor


def query_aware_kscore_slot(Q_slot, dK_slot, A_slot, V_slot, head_dim):
    """Phase-3 query-aware K-side residual score for one K slot (gated by
    CAREKV_KSCORE_LIVE=1; OFF by default → existing proxy K-score is unchanged).

      ΔK = K_fp − K_quant  (== the stored K residual R_K for this slot)
      ΔS = Q · ΔKᵀ / √head_dim                         (score perturbation)
      sensitivity = A · (1 − A)                         (softmax sensitivity proxy)
      KScore[q] = ‖ΔS·sensitivity‖_tokens · ‖V contribution‖

    Shapes: Q_slot (Q, cg), dK_slot (nv, cg), A_slot (Q, nv), V_slot (nv, D).
    Returns (Q,) per-query KScore — finite, non-negative.
    """
    dS = (Q_slot.float() @ dK_slot.float().T) / math.sqrt(max(head_dim, 1))   # (Q, nv)
    sens = A_slot.float() * (1.0 - A_slot.float())                            # (Q, nv)
    k_perturb = (dS * sens).norm(dim=1)                                       # (Q,)
    v_contrib = V_slot.float().norm()                                         # scalar (slot V magnitude)
    out = (k_perturb * v_contrib)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


# ─────────────────────────────────────────────
# Module-level slot-correction helper
# ─────────────────────────────────────────────

def _page_token_offsets(
    cache: CAREKVCache,
    layer_id: int,
    kv_head: int,
    page_ids: List[int],
) -> Dict[int, Tuple[int, int]]:
    """Return {page_id: (start, end)} into the concat-valid attention vector."""
    offsets: Dict[int, Tuple[int, int]] = {}
    cursor = 0
    for pid in page_ids:
        n = int(cache.valid_tokens[layer_id, kv_head, pid].item())
        if n > 0:
            offsets[pid] = (cursor, cursor + n)
        cursor += n
    return offsets


def apply_slot_corrections(
    cache: CAREKVCache,
    cfg: CacheConfig,
    router: ResidualRouter,
    layer_id: int,
    kv_head: int,
    page_ids: List[int],
    q: Tensor,             # (D,) float, post-RoPE query for current step
    s_base: Tensor,        # (N_valid,) float — base attention logits
    a_base: Tensor,        # (N_valid,) float — base attention weights
    V_base: Tensor,        # (N_valid, D) float — dequantized base V (concat-valid)
    O_base: Tensor,        # (D,) float — base attention output
    kind: str = "both",
    k_corr_scale: float = 1.0,
    score_normalize: bool = False,    # passed through to router.route for
                                       # kind=="both" per-kind score scaling
    debug_stats: Optional[Dict[str, Any]] = None,
    slot_cache: Optional[Dict[Any, Tensor]] = None,    # P4-cached: pre-unpacked
                                                       # residuals keyed by
                                                       # (kind, page_id, group_idx).
) -> Tensor:
    """
    Returns ΔO of shape (D,) — the residual correction to add to O_base.

    Reads only from stored slots in `cache`; never touches a full R_K/R_V.
    """
    if kind not in {"v", "k", "both"}:
        raise ValueError(f"Unknown kind={kind}")

    D = cfg.head_dim
    device = O_base.device
    delta_V = torch.zeros(D, device=device, dtype=torch.float32)
    delta_K = torch.zeros(D, device=device, dtype=torch.float32)

    mode = k_correction_mode()
    N_valid = a_base.shape[0]
    if mode == "exact":
        ds_full = torch.zeros(N_valid, device=device, dtype=torch.float32)
        RV_full = torch.zeros(N_valid, D, device=device, dtype=torch.float32)
        sel_v_tok = torch.zeros(N_valid, device=device, dtype=torch.bool)

    # Route stored slots under read budget (kind-aware: budget applies only
    # to the requested slot type(s), so V-only doesn't waste budget on K and
    # vice versa).
    k_selected, v_selected = router.route(
        cache=cache,
        kv_head=kv_head,
        page_ids=page_ids,
        q=q,
        s_base=s_base,
        a_base=a_base,
        O_base=O_base,
        V_base_full=V_base,
        kind=kind,
        score_normalize=score_normalize,
        debug_stats=debug_stats,
    )

    # Per-page (start, end) offsets into a_base / V_base.
    offsets = _page_token_offsets(cache, layer_id, kv_head, page_ids)

    # ── V correction: ΔO_V = Σ_t a_t · R_V,t ────────────────────────
    n_v_applied = 0
    for (page_id, vb_idx, slot) in v_selected:
        if page_id not in offsets:
            continue
        p_start, p_end = offsets[page_id]
        n_valid_in_page = p_end - p_start

        t0 = vb_idx * cfg.v_token_block
        t1_padded = t0 + cfg.v_token_block
        # Slot was packed at fixed block size; trim to valid rows in this block.
        t1_valid = min(t1_padded, n_valid_in_page)
        if t1_valid <= t0:
            continue
        blk_size = t1_valid - t0

        # Cached path (P4): if a pre-unpacked tensor is provided by the
        # caller for this slot, reuse it instead of re-unpacking 4-bit.
        cache_key = ("V", page_id, vb_idx)
        if slot_cache is not None and cache_key in slot_cache:
            R_V_blk_full = slot_cache[cache_key]
        else:
            packed, scale = cache.read_v_residual(slot)
            numel_full = cfg.v_token_block * D
            R_V_blk_full = unpack_residual(packed, scale, numel_full, cfg.residual_bits).reshape(
                cfg.v_token_block, D
            ).to(device)
            if slot_cache is not None:
                slot_cache[cache_key] = R_V_blk_full
        R_V_blk = R_V_blk_full[:blk_size]

        blk_a = a_base[p_start + t0 : p_start + t1_valid]
        delta_V += (blk_a.unsqueeze(-1) * R_V_blk).sum(0)
        if mode == "exact":
            RV_full[p_start + t0 : p_start + t1_valid] = R_V_blk.float()
            sel_v_tok[p_start + t0 : p_start + t1_valid] = True
        n_v_applied += 1

    # ── K correction: ΔO_K ≈ Σ_t a_t · (q · R_K,t) · (V_t − O_base) ──
    n_k_applied = 0
    q_f = q.float()
    for (page_id, cg_idx, slot) in k_selected:
        if page_id not in offsets:
            continue
        p_start, p_end = offsets[page_id]
        n_valid_in_page = p_end - p_start

        c0 = cg_idx * cfg.k_channel_group
        c1 = c0 + cfg.k_channel_group
        cache_key = ("K", page_id, cg_idx)
        if slot_cache is not None and cache_key in slot_cache:
            R_K_blk_full = slot_cache[cache_key]
        else:
            packed, scale = cache.read_k_residual(slot)
            numel_full = cfg.page_size * cfg.k_channel_group
            R_K_blk_full = unpack_residual(packed, scale, numel_full, cfg.residual_bits).reshape(
                cfg.page_size, cfg.k_channel_group
            ).to(device)
            if slot_cache is not None:
                slot_cache[cache_key] = R_K_blk_full
        R_K_blk = R_K_blk_full[:n_valid_in_page]      # (n_valid, cg_size)

        page_a = a_base[p_start:p_end]                  # (n_valid,)
        page_V = V_base[p_start:p_end]                  # (n_valid, D)

        q_cg = q_f[c0:c1]                                # (cg_size,)
        qdot_rk = (R_K_blk * q_cg.unsqueeze(0)).sum(-1) # (n_valid,)
        V_diff = page_V - O_base.unsqueeze(0)            # (n_valid, D)

        weights = page_a * qdot_rk
        delta_K += (weights.unsqueeze(-1) * V_diff).sum(0)
        if mode == "exact":
            ds_full[p_start:p_end] += qdot_rk.float()
        n_k_applied += 1

    # See vectorized_joint_correction: with no K slots read the exact form is
    # algebraically the linear V term, so fall through and keep it bit-identical.
    if mode == "exact" and n_k_applied > 0:
        delta_O = exact_softmax_correction(
            A=a_base.unsqueeze(0),
            ds=(ds_full / math.sqrt(D)).unsqueeze(0),
            V_base=V_base,
            O_base=O_base.unsqueeze(0),
            RV_full=RV_full,
            sel_tok=sel_v_tok.unsqueeze(0),
        ).squeeze(0)
    else:
        delta_O = delta_V + k_corr_scale * delta_K

    if debug_stats is not None:
        debug_stats.setdefault("v_slots_read", 0)
        debug_stats.setdefault("k_slots_read", 0)
        debug_stats.setdefault("delta_v_norm_sum", 0.0)
        debug_stats.setdefault("delta_k_norm_sum", 0.0)
        debug_stats.setdefault("delta_o_norm_sum", 0.0)
        debug_stats.setdefault("o_base_norm_sum", 0.0)
        debug_stats.setdefault("n_queries", 0)
        debug_stats["v_slots_read"] += n_v_applied
        debug_stats["k_slots_read"] += n_k_applied
        debug_stats["delta_v_norm_sum"] += float(delta_V.norm().item())
        debug_stats["delta_k_norm_sum"] += float((k_corr_scale * delta_K).norm().item())
        debug_stats["delta_o_norm_sum"] += float(delta_O.norm().item())
        debug_stats["o_base_norm_sum"] += float(O_base.norm().item())
        debug_stats["n_queries"] += 1

    return delta_O.to(O_base.dtype)


# ─────────────────────────────────────────────
# Vectorized V correction (P4-vectorized)
#
# Builds per-(layer, kv_head) tensors that index each stored V slot once,
# then computes V scores, topk, and ΔO_V via batched torch ops over all
# T query tokens at once.  K correction stays on the cached path (hybrid).
# Bit-equivalent to the cached path when read_budget covers ≥ topk.
# ─────────────────────────────────────────────

def _build_v_slot_index(cache: CAREKVCache, cfg: CacheConfig,
                        layer_id: int, kv_head: int, page_ids: List[int],
                        device, dtype):
    """One-time scan of a (layer, kv_head) cache.  Returns:
        slot_v_resid       : (N_v, v_token_block, D) float32 — unpacked
        slot_token_starts  : (N_v,) long — start of each slot's tokens in
                              the concat-valid attention vector
        slot_token_lens    : (N_v,) long — number of *valid* tokens in
                              each slot (≤ v_token_block)
        slot_v_err         : (N_v,) float32 — per-slot err_norm × sensitivity
        slot_ids           : list of (page_id, vb_idx, slot_id) for tracing
    """
    cfg_v_block = cfg.v_token_block
    D = cfg.head_dim
    sens = cfg.layer_sensitivity[layer_id] if cfg.layer_sensitivity else 1.0

    resids: List[Tensor] = []
    starts: List[int] = []
    lens: List[int] = []
    errs: List[float] = []
    ids: List[Tuple[int, int, int]] = []

    cursor = 0
    for pid in page_ids:
        meta: Optional[PageMeta] = cache.meta_table[layer_id][kv_head][pid]
        n_valid = int(cache.valid_tokens[layer_id, kv_head, pid].item())
        if n_valid == 0 or meta is None:
            continue
        for vb, slot in enumerate(meta.v_residual_slots):
            if slot < 0:
                continue
            t0 = vb * cfg_v_block
            t1_valid = min(t0 + cfg_v_block, n_valid)
            if t1_valid <= t0:
                continue
            packed, scale = cache.read_v_residual(slot)
            numel_full = cfg_v_block * D
            r = unpack_residual(packed, scale, numel_full, cfg.residual_bits).reshape(
                cfg_v_block, D
            ).to(device)
            resids.append(r)
            starts.append(cursor + t0)
            lens.append(t1_valid - t0)
            err = float(meta.v_error_norm[vb].item()
                        if meta.v_error_norm is not None else 1.0)
            errs.append(err * sens)
            ids.append((pid, vb, slot))
        cursor += n_valid

    if not resids:
        return None
    slot_v_resid = torch.stack(resids, dim=0).to(device=device, dtype=torch.float32)
    slot_token_starts = torch.tensor(starts, dtype=torch.long, device=device)
    slot_token_lens = torch.tensor(lens, dtype=torch.long, device=device)
    slot_v_err = torch.tensor(errs, dtype=torch.float32, device=device)
    return slot_v_resid, slot_token_starts, slot_token_lens, slot_v_err, ids


def vectorized_v_correction(
    cache: CAREKVCache, cfg: CacheConfig,
    layer_id: int, kv_head: int, page_ids: List[int],
    A: Tensor,           # (T, N_total) attention weights (already softmax)
    N_total: int,
    D: int,
    debug_stats: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Tensor], int]:
    """Returns (delta_V[T, D] or None if no V slots, n_reads_total).

    Reuses the router's V scoring formula (`blk_attn_mass × v_err_norm ×
    sensitivity`) and the absolute/ratio read-budget resolver via
    `read_abs_v` / `read_budget_ratio_v` / `read_budget_ratio`.
    """
    idx = _build_v_slot_index(cache, cfg, layer_id, kv_head, page_ids,
                              device=A.device, dtype=A.dtype)
    if idx is None:
        return None, 0
    V_resid, starts, lens, v_err, slot_ids = idx
    N_v = V_resid.shape[0]
    T = A.shape[0]

    # ── 1) Per-(t, s) attention mass over slot tokens ─────────────
    # Build (N_v, N_total) 0/1 mask in one vectorized expansion.
    arange_n = torch.arange(N_total, device=A.device).unsqueeze(0)   # (1, N_total)
    ends = starts + lens
    slot_mask = (arange_n >= starts.unsqueeze(1)) & (arange_n < ends.unsqueeze(1))
    slot_mask = slot_mask.to(A.dtype)                                # (N_v, N_total)

    blk_attn_mass = A @ slot_mask.T                                  # (T, N_v)
    scores = blk_attn_mass * v_err.unsqueeze(0).to(A.dtype)          # (T, N_v)

    # ── 2) Resolve V budget (per-layer multiplier applied here too) ──
    from .cache import layer_budget_multiplier
    lm = layer_budget_multiplier(cfg, layer_id)
    if cfg.read_budget_mode == "absolute":
        budget_v = int(round(max(0, int(cfg.read_abs_v)) * lm))
    else:
        rv_override = cfg.read_budget_ratio_v
        rv = cfg.read_budget_ratio if (rv_override is None or rv_override < 0) else rv_override
        rv *= lm
        budget_v = max(1, int(N_v * rv)) if (rv > 0 and N_v > 0) else 0
    budget_v = min(budget_v, N_v)
    if budget_v <= 0:
        return None, 0

    # ── 3) Per-query top-budget_v selection ───────────────────────
    topk_idx = torch.topk(scores, k=budget_v, dim=-1).indices       # (T, budget_v)
    selected = torch.zeros(T, N_v, dtype=torch.bool, device=A.device)
    selected.scatter_(1, topk_idx, True)

    # ── 4) Build per-token map: token → (slot, local_pos) ─────────
    # Non-overlapping V slots ⇒ each valid token is owned by at most 1 slot.
    in_slot = slot_mask.bool()                                       # (N_v, N_total)
    covered = in_slot.any(dim=0)                                     # (N_total,)
    if not covered.any():
        return None, 0
    token_to_slot = in_slot.long().argmax(dim=0)                     # (N_total,)
    # local_pos[j] = j - starts[token_to_slot[j]]
    local_pos = (arange_n.squeeze(0) - starts[token_to_slot]).clamp(min=0)

    # ── 5) Gather per-token V residual: (N_total, D) ──────────────
    V_resid_full = torch.zeros(N_total, D, device=A.device, dtype=torch.float32)
    cov_idx = covered.nonzero(as_tuple=True)[0]
    V_resid_full[cov_idx] = V_resid[token_to_slot[cov_idx], local_pos[cov_idx]]

    # ── 6) Per-(t, j) selection mask ──────────────────────────────
    safe_t2s = token_to_slot.clamp(min=0)
    sel_tok = selected[:, safe_t2s] & covered.unsqueeze(0)          # (T, N_total)

    weighted_a = A.to(torch.float32) * sel_tok.to(torch.float32)    # (T, N_total)
    delta_V = weighted_a @ V_resid_full                              # (T, D)

    n_reads = int(selected.sum().item())
    if debug_stats is not None:
        debug_stats["v_slots_read"] = debug_stats.get("v_slots_read", 0) + n_reads
        debug_stats["delta_v_norm_sum"] = debug_stats.get("delta_v_norm_sum", 0.0) + \
            float(delta_V.norm(dim=-1).sum().item())
    return delta_V, n_reads


# ─────────────────────────────────────────────
# Vectorized joint K+V correction (P5 — joint+both unblock)
#
# Replaces the per-(query_head, token) router+correction Python loop with a
# single batched path over all Q = (kv_group × T) queries of a kv_head.
# Reproduces router.route() scoring + the joint/separate selection policy and
# apply_slot_corrections() math, but with torch ops + torch.topk.
#
# Math preserved exactly (fp32 throughout, matching the cached path):
#   ΔO_V[q] = Σ_{sel V-slot} Σ_{t∈block} A[q,t]·R_V[slot,t]
#   ΔO_K[q] = Σ_t A[q,t]·wK[q,t]·(V_t − O_base[q])
#           = (A·wK) @ V_base − rowsum(A·wK)·O_base
#   where wK[q,t] = Σ_{sel cg of t's page} q[q,c0:c1]·R_K[cg,t]
# ─────────────────────────────────────────────

def _build_k_slot_index(cache, cfg, layer_id, kv_head, page_ids, device):
    """Per (layer, kv_head): unpack every stored K residual slot once.
    Returns lists aligned by slot index:
      R_K   : (n_valid, k_channel_group) fp32 residual
      cg    : channel-group index (→ c0:c1 channels)
      sketch: (sketch_dim,) fp32 — meta.k_sketch[cg]
      pstart: token start of the slot's page in the concat-valid vector
      nvalid: valid tokens in the page
      ids   : (page_id, cg, slot_id)
    """
    R_Ks, cgs, sketches, pstarts, nvalids, ids = [], [], [], [], [], []
    cursor = 0
    for pid in page_ids:
        meta = cache.meta_table[layer_id][kv_head][pid]
        n_valid = int(cache.valid_tokens[layer_id, kv_head, pid].item())
        if n_valid == 0 or meta is None:
            cursor += n_valid
            continue
        if meta.k_sketch is not None:
            for cg, slot in enumerate(meta.k_residual_slots):
                if slot < 0:
                    continue
                packed, scale = cache.read_k_residual(slot)
                numel_full = cfg.page_size * cfg.k_channel_group
                r_full = unpack_residual(packed, scale, numel_full, cfg.residual_bits).reshape(
                    cfg.page_size, cfg.k_channel_group).to(device)
                R_Ks.append(r_full[:n_valid].float())
                cgs.append(cg)
                sketches.append(meta.k_sketch[cg].float().to(device))
                pstarts.append(cursor)
                nvalids.append(n_valid)
                ids.append((pid, cg, slot))
        cursor += n_valid
    return R_Ks, cgs, sketches, pstarts, nvalids, ids


def vectorized_joint_correction(
    cache, cfg, router, layer_id, kv_head, page_ids,
    Q_q: Tensor,        # (Q, D) post-RoPE queries
    S: Tensor,          # (Q, N) base logits over valid tokens
    A: Tensor,          # (Q, N) base attention weights (softmax)
    V_base: Tensor,     # (N, D) dequantized base V (concat-valid)
    O_base: Tensor,     # (Q, D) base attention outputs
    kind: str,
    k_corr_scale: float,
    score_normalize: bool,
    debug_stats: Optional[Dict[str, Any]] = None,
    _no_chunk: bool = False,
    _q_offset: int = 0,
) -> Tensor:
    """Batched ΔO[(Q, D)] for all queries of one kv_head. Mirrors
    apply_slot_corrections(kind, policy) + router.route() exactly in fp32."""
    device = Q_q.device
    Qn, D = Q_q.shape
    N = A.shape[1]

    # ── Chunked correction (CAREKV_CHUNKED_CORRECTION=1): process queries in
    # chunks of CAREKV_CHUNK_SIZE to cap the peak memory of the per-query scoring
    # tensors ((chunk,N) and (chunk,n_v,D)), which is what OOMs at SL≥512 on 7B.
    # OFF by default.
    #
    # Exactness (isolated by tools/debug_chunked_carekv_equivalence.py):
    #   - The correction is ALGORITHMICALLY per-query — layer-0 delta is bit-exact
    #     for any chunk size, and READ0 (budget 0) is always bit-exact.
    #   - With TF32 OFF + deterministic algorithms, chunk_size=128 (Qn=256) is
    #     BIT-IDENTICAL to full (scores, selected indices, and delta all 0 diff).
    #     The earlier "~0.05% approximate" reading was a **TF32 artifact** (TF32
    #     matmuls round differently per batch size).
    #   - At smaller chunks the matmul tiling for small M differs, producing a
    #     ~1e-8 float-accumulation-order difference at layer 1 that PROPAGATES and
    #     amplifies through the deep network, eventually flipping near-tie top-k
    #     selections (≤0.05% PPL). This is the same class of numerical variation
    #     as TF32-on/off or fp16/bf16 — not a logic error.
    # → Paper-usable under TF32-off determinism with a chunk size large enough for
    #   tiling to align (bit-exact); otherwise numerically-equivalent within
    #   float-accumulation order. Prefer larger chunks; keep TF32 off.
    if (not _no_chunk and Qn > 0 and N > 0
            and os.environ.get("CAREKV_CHUNKED_CORRECTION", "0") == "1"):
        # CAREKV_SCORE_CHUNK_SIZE is honored as an alias (query chunking caps the
        # score-tensor memory, so the two are the same knob here).
        cs = int(os.environ.get("CAREKV_CHUNK_SIZE")
                 or os.environ.get("CAREKV_SCORE_CHUNK_SIZE") or 128)
        if cs > 0 and Qn > cs:
            out = torch.zeros(Qn, D, device=device, dtype=torch.float32)
            for c0 in range(0, Qn, cs):
                c1 = min(c0 + cs, Qn)
                out[c0:c1] = vectorized_joint_correction(
                    cache, cfg, router, layer_id, kv_head, page_ids,
                    Q_q[c0:c1], S[c0:c1], A[c0:c1], V_base, O_base[c0:c1],
                    kind, k_corr_scale, score_normalize, debug_stats,
                    _no_chunk=True, _q_offset=c0)  # per-chunk records captured
            if _CHUNK_REC is not None:
                _CHUNK_REC.append(dict(layer_id=layer_id, kv_head=kv_head, q_offset=-1,
                                       Qn=Qn, N=N, delta=out.detach().clone(),
                                       score_V=None, sel_v=None, score_K=None, sel_k=None,
                                       budget_v=-1, budget_k=-1, assembled=True))
            return out

    sens = float(router.sensitivity)
    policy = getattr(cfg, "route_policy", "separate")
    A = A.float(); S = S.float(); V_base = V_base.float(); O_base = O_base.float()
    delta = torch.zeros(Qn, D, device=device, dtype=torch.float32)
    if N == 0:
        return delta

    # ── page token ranges (start, n_valid) in concat-valid order ──
    page_ranges = []
    cursor = 0
    for pid in page_ids:
        n = int(cache.valid_tokens[layer_id, kv_head, pid].item())
        if n > 0:
            page_ranges.append((pid, cursor, n))
        cursor += n

    # ── V slot index (reuse builder) ──
    v_idx = _build_v_slot_index(cache, cfg, layer_id, kv_head, page_ids, device, A.dtype)
    # ── K slot index ──
    R_Ks, k_cgs, k_sketches, k_pstarts, k_nvalids, k_ids = (
        _build_k_slot_index(cache, cfg, layer_id, kv_head, page_ids, device))
    n_k = len(R_Ks)
    n_v = 0 if v_idx is None else v_idx[0].shape[0]

    want_k = kind in {"k", "both"} and n_k > 0
    want_v = kind in {"v", "both"} and n_v > 0
    if not want_k and not want_v:
        return delta

    budget_k, budget_v = _resolve_read_budgets(
        cfg, kind, n_k if want_k else 0, n_v if want_v else 0, layer_id=layer_id)
    if budget_k == 0 and budget_v == 0:
        return delta

    # ── K scoring ──
    score_K = None
    pf_keep_k = None   # router pre-filter shortlist mask (applied post-normalize)
    # Phase-3 query-aware K-score (gated). CAREKV_KSCORE_LIVE=1 replaces the proxy
    # score_K with the exact query-aware K-score (ΔS·sensitivity · ‖V‖). OFF by
    # default → the block below is byte-identical to the current path.
    _kscore_live = os.environ.get("CAREKV_KSCORE_LIVE", "0") == "1"
    if want_k and budget_k > 0:
        s_top = S.max(dim=1, keepdim=True).values                      # (Q,1)
        inv_margin = 1.0 / (s_top - S).clamp(min=EPS)                  # (Q,N)
        # V-diff norms ||V_t - O_base[q]|| : (Q, N)
        vdiff = (V_base.unsqueeze(0) - O_base.unsqueeze(1)).norm(dim=-1)  # (Q,N)
        Aw = A
        # sketch projection for |q·R_K| estimate
        from .residual_store import get_sketch_proj
        P = get_sketch_proj(cfg.head_dim, cfg.k_channel_group, cfg.sketch_dim, device)
        score_K = torch.zeros(Qn, n_k, device=device, dtype=torch.float32)
        # Router O(S) scoring-BW pre-filter (router_prefilter_bw_design.md,
        # router_sign_prefilter_design.md). C>0: score-mask so only the top-C K
        # slots per query (by a cheap proxy) can be selected → the exact sketch
        # score is needed for C, not all n_k, slots. C=0 → exact.
        #   SIGN_PREFILTER_B>0 → directional sign-sketch (SimHash) proxy: rank by
        #     ‖q_sketch‖·‖k_sketch‖·|cos θ̂| with θ̂ from the b-bit sign planes —
        #     recovers the direction the magnitude bound dropped (b/8+2 B/cand).
        #   SIGN_PREFILTER_B=0 → v1 magnitude bound ‖q‖·‖R_K‖ (direction deleted).
        prefilter_c = int(os.environ.get("CAREKV_ROUTER_PREFILTER_C", "0") or "0")
        sign_b = int(os.environ.get("CAREKV_ROUTER_SIGN_PREFILTER_B", "0") or "0")
        _use_pf = 0 < prefilter_c < n_k and not _kscore_live
        if _use_pf:
            qnorm_v = Q_q.float().norm(dim=1)                          # (Q,)
            ub_K = torch.zeros(Qn, n_k, device=device, dtype=torch.float32)
            _sb = cfg.sketch_dim if sign_b <= 0 else min(sign_b, cfg.sketch_dim)
        for si in range(n_k):
            c0 = k_cgs[si] * cfg.k_channel_group
            c1 = c0 + cfg.k_channel_group
            ps, nv = k_pstarts[si], k_nvalids[si]
            if _kscore_live:
                try:
                    ks = query_aware_kscore_slot(
                        Q_q[:, c0:c1], R_Ks[si], A[:, ps:ps + nv], V_base[ps:ps + nv], cfg.head_dim)
                    if not torch.isfinite(ks).all():
                        raise ValueError("non-finite kscore")
                    score_K[:, si] = ks * sens
                    continue
                except Exception:
                    pass  # fall through to the proxy on any failure
            page_a = A[:, ps:ps + nv]                                  # (Q, nv)
            page_attn_mass = page_a.sum(dim=1)                        # (Q,)
            boundary = (page_a * inv_margin[:, ps:ps + nv]).sum(dim=1)  # (Q,)
            page_vdiff = (page_a * vdiff[:, ps:ps + nv]).sum(dim=1)    # (Q,)
            q_sketch = Q_q[:, c0:c1].float() @ P                       # (Q, sketch_dim)
            qdotr = (q_sketch * k_sketches[si].unsqueeze(0)).sum(dim=1).abs()  # (Q,)
            score_K[:, si] = (page_attn_mass * qdotr * boundary
                              * page_vdiff * sens)
            if _use_pf:
                if sign_b > 0:
                    # directional sign-sketch proxy: |q·R_K| ≈ ‖a‖‖b‖·|cos θ̂|,
                    # θ̂ from the b-bit sign planes of q_sketch (a) & k_sketch (b).
                    qs_sign = q_sketch[:, :_sb] > 0                    # (Q, b)
                    ks_sign = (k_sketches[si][:_sb] > 0).unsqueeze(0)  # (1, b)
                    ham = (qs_sign != ks_sign).sum(dim=1).float()      # (Q,)
                    cos_est = torch.cos(math.pi * ham / _sb).abs()     # (Q,)
                    dhat = (q_sketch.norm(dim=1)
                            * k_sketches[si].norm() * cos_est)         # (Q,)
                    ub_K[:, si] = (page_attn_mass * dhat * boundary
                                   * page_vdiff * sens)
                else:
                    rk_norm = R_Ks[si].float().norm()
                    ub_K[:, si] = (page_attn_mass * qnorm_v * rk_norm
                                   * boundary * page_vdiff * sens)
        if _use_pf:
            keep = torch.topk(ub_K, prefilter_c, dim=1).indices        # (Q, C)
            # Restrict *selection* to the shortlist, but keep score_K intact so
            # the per-kind normalization mean below is not polluted by -inf. The
            # mask is applied to score_K_r (post-normalize) right before top-k.
            pf_keep_k = torch.zeros_like(score_K, dtype=torch.bool)
            pf_keep_k.scatter_(1, keep, True)
            if debug_stats is not None:
                debug_stats["k_prefilter_pool"] = (
                    debug_stats.get("k_prefilter_pool", 0) + Qn * n_k)
                debug_stats["k_prefilter_scored"] = (
                    debug_stats.get("k_prefilter_scored", 0) + Qn * prefilter_c)
                # analytical Stage-1 O(S) bytes/candidate: sign bits + fp16 norm
                debug_stats["k_stage1_bytes_per_cand"] = (
                    (_sb / 8.0 + 2.0) if sign_b > 0 else 2.0)
    # ── V scoring ──
    score_V = None
    if want_v and budget_v > 0:
        V_resid, v_starts, v_lens, v_err, _vids = v_idx
        arange_n = torch.arange(N, device=device).unsqueeze(0)
        v_mask = ((arange_n >= v_starts.unsqueeze(1))
                  & (arange_n < (v_starts + v_lens).unsqueeze(1))).float()  # (n_v,N)
        _rscore = os.environ.get("CAREKV_RESIDUAL_SCORE", "raw_error")
        if _rscore == "attention_output":
            # Exact attention-output impact (spec §B): score_V[q,b] =
            # ‖ Σ_{t∈block b} A[q,t] · (V_fp16−V_quant)[t] ‖ — the L2 norm of
            # the slot's actual contribution to the attention output, not the
            # mass×norm proxy. Gate-safe: scoring only affects WHICH slots are
            # read when budget>0 (READ0 / fp-mode never reach here). Falls back
            # to the raw-error proxy on any non-finite result.
            try:
                vtb = V_resid.shape[1]
                jidx = torch.arange(vtb, device=device)
                col = (v_starts.unsqueeze(1) + jidx.unsqueeze(0)).clamp(0, N - 1)  # (n_v,vtb)
                valid = (jidx.unsqueeze(0) < v_lens.unsqueeze(1)).to(A.dtype)      # (n_v,vtb)
                A_slot = A[:, col] * valid.unsqueeze(0)                            # (Q,n_v,vtb)
                contrib = torch.einsum("qbj,bjd->qbd", A_slot, V_resid.float())    # (Q,n_v,D)
                score_V = contrib.norm(dim=2)                                      # (Q,n_v)
                score_V = torch.nan_to_num(score_V, nan=0.0, posinf=0.0, neginf=0.0)
                if not torch.isfinite(score_V).all():
                    raise ValueError("non-finite attention_output V score")
                score_V = score_V * sens
            except Exception:
                blk_attn_mass = A @ v_mask.T
                score_V = blk_attn_mass * v_err.unsqueeze(0).float() * sens
        else:
            blk_attn_mass = A @ v_mask.T                               # (Q, n_v)
            score_V = blk_attn_mass * v_err.unsqueeze(0).float()       # sens folded below
            score_V = score_V * sens

    # ── Phase 11B (EXPERIMENTAL, gated): selector-variant + position-policy
    #    re-score in the vectorized path, mirroring the residual_store /
    #    residual_router hooks so the SAME env vars drive both the route path
    #    and this fast path. Default (current / none) → no-op (byte-identical). ──
    _ph11b_sv = os.environ.get("CAREKV_SELECTOR_VARIANT", "current")
    _ph11b_pp = os.environ.get("CAREKV_POSITION_POLICY", "none")
    if _ph11b_sv != "current" or _ph11b_pp != "none":
        # selector variant overrides the base score formula (read-side selection)
        if _ph11b_sv == "random":
            if score_V is not None:
                score_V = torch.rand_like(score_V)
            if score_K is not None:
                score_K = torch.rand_like(score_K)
        elif _ph11b_sv == "oracle_residual_magnitude":
            if score_V is not None:                       # rank V by raw residual norm
                score_V = v_err.unsqueeze(0).float().expand(Qn, -1).contiguous()
            if score_K is not None and n_k > 0:           # rank K by raw slot norm
                _knorm = torch.stack([R_Ks[si].norm() for si in range(n_k)]).float()
                score_K = _knorm.unsqueeze(0).expand(Qn, -1).contiguous()
        elif _ph11b_sv == "oracle_reconstruction_error":
            if score_V is not None:                       # exact ‖Σ_t A·R_V‖ impact
                _vtb = V_resid.shape[1]
                _j = torch.arange(_vtb, device=device)
                _col = (v_starts.unsqueeze(1) + _j.unsqueeze(0)).clamp(0, N - 1)
                _val = (_j.unsqueeze(0) < v_lens.unsqueeze(1)).to(A.dtype)
                _As = A[:, _col] * _val.unsqueeze(0)
                _contrib = torch.einsum("qbj,bjd->qbd", _As, V_resid.float())
                score_V = torch.nan_to_num(_contrib.norm(dim=2), nan=0.0,
                                           posinf=0.0, neginf=0.0) * sens
            # K kept query-aware (already reconstruction-aware)
        # position policy multiplies V scores by a per-slot position weight; uses
        # the actual valid length N as seq_len (full-prefill = whole sequence).
        if _ph11b_pp != "none" and score_V is not None and N > 0:
            from .residual_store import _position_weight as _pw
            _wts = torch.tensor([_pw(_ph11b_pp, int(t.item()), N) for t in v_starts],
                                device=device, dtype=score_V.dtype)

    # ── per-query normalization (kind==both only), matching _norm ──
    if score_normalize and kind == "both" and score_K is not None and score_V is not None:
        nk = score_K.abs().mean(dim=1, keepdim=True).clamp(min=EPS)
        nv = score_V.abs().mean(dim=1, keepdim=True).clamp(min=EPS)
        score_K_r = score_K / nk
        score_V_r = score_V / nv
        # Phase-3 combined_kvscore (gated): CombinedScore = λ_k·norm(KScore) +
        # λ_v·norm(VScore). Default λ=1/1 == the existing equal-weight joint, so
        # behaviour is unchanged unless the lambdas are set. OFF unless live.
        if _kscore_live:
            lk = float(os.environ.get("CAREKV_KSCORE_LAMBDA_K", "1") or 1)
            lv = float(os.environ.get("CAREKV_KSCORE_LAMBDA_V", "1") or 1)
            score_K_r = score_K_r * lk
            score_V_r = score_V_r * lv
    else:
        score_K_r, score_V_r = score_K, score_V

    # Apply the router pre-filter shortlist AFTER normalization: only the top-C
    # K slots (by the cheap proxy) remain eligible for selection. Done here (not
    # on score_K) so the normalization mean above is computed on true scores.
    if pf_keep_k is not None and score_K_r is not None:
        score_K_r = score_K_r.masked_fill(~pf_keep_k, float("-inf"))

    # ── selection masks per query ──
    sel_k = torch.zeros(Qn, n_k, dtype=torch.bool, device=device) if score_K is not None else None
    sel_v = torch.zeros(Qn, n_v, dtype=torch.bool, device=device) if score_V is not None else None

    if policy == "joint" and kind == "both" and score_K_r is not None and score_V_r is not None:
        merged_budget = (budget_k + budget_v) if (budget_k > 0 and budget_v > 0) \
            else max(budget_k, budget_v)
        merged = torch.cat([score_K_r, score_V_r], dim=1)             # (Q, n_k+n_v)
        mb = min(merged_budget, merged.shape[1])
        top = torch.topk(merged, k=mb, dim=1).indices                # (Q, mb)
        sel_merged = torch.zeros(Qn, n_k + n_v, dtype=torch.bool, device=device)
        sel_merged.scatter_(1, top, True)
        sel_k = sel_merged[:, :n_k]
        sel_v = sel_merged[:, n_k:]
    else:
        # separate (default): independent per-kind top-k
        if score_K_r is not None and budget_k > 0:
            kb = min(budget_k, n_k)
            top = torch.topk(score_K_r, k=kb, dim=1).indices
            sel_k.scatter_(1, top, True)
        if score_V_r is not None and budget_v > 0:
            vb = min(budget_v, n_v)
            top = torch.topk(score_V_r, k=vb, dim=1).indices
            sel_v.scatter_(1, top, True)

    # ── K apply: build wK[Q,N], then delta_K = (A·wK)@V − rowsum(A·wK)·O ──
    # Optional bounded K-stabilization knobs (default OFF → LLaMA path unchanged):
    #   CAREKV_K_QDOTR_CLAMP_PCT : clamp |q·R_K| to its pXX percentile (K_clipped)
    #   CAREKV_K_NORM_GUARD_PCT  : skip K slots whose residual norm exceeds the
    #                              pXX percentile of slot norms (K_norm_guard)
    # These dampen the 1st-order Jacobian blow-up on outlier-heavy K (e.g. Qwen).
    import os as _os
    _qclamp = float(_os.environ.get("CAREKV_K_QDOTR_CLAMP_PCT", "0") or 0)
    _nguard = float(_os.environ.get("CAREKV_K_NORM_GUARD_PCT", "0") or 0)
    mode = k_correction_mode()
    wK = None
    n_k_reads = 0
    if sel_k is not None and sel_k.any():
        # K_norm_guard: zero out selection for slots with too-large residual norm.
        if _nguard > 0 and n_k > 0:
            slot_norms = torch.stack([R_Ks[si].norm() for si in range(n_k)])
            thr_n = torch.quantile(slot_norms, min(max(_nguard / 100.0, 0.0), 1.0))
            for si in range(n_k):
                if slot_norms[si] > thr_n:
                    sel_k[:, si] = False
        wK = torch.zeros(Qn, N, device=device, dtype=torch.float32)
        for si in range(n_k):
            c0 = k_cgs[si] * cfg.k_channel_group
            c1 = c0 + cfg.k_channel_group
            ps, nv = k_pstarts[si], k_nvalids[si]
            qdotr_full = Q_q[:, c0:c1].float() @ R_Ks[si].T            # (Q, nv)
            if _qclamp > 0 and qdotr_full.numel() > 0:
                thr = torch.quantile(qdotr_full.abs().flatten(),
                                     min(max(_qclamp / 100.0, 0.0), 1.0))
                qdotr_full = qdotr_full.clamp(-thr, thr)
            wK[:, ps:ps + nv] += sel_k[:, si].float().unsqueeze(1) * qdotr_full
        n_k_reads = int(sel_k.sum().item())

    # ── V apply ──
    V_resid_full = None
    sel_tok = None
    n_v_reads = 0
    if sel_v is not None and sel_v.any():
        V_resid, v_starts, v_lens, v_err, _vids = v_idx
        arange_n = torch.arange(N, device=device).unsqueeze(0)
        in_slot = ((arange_n >= v_starts.unsqueeze(1))
                   & (arange_n < (v_starts + v_lens).unsqueeze(1)))    # (n_v,N)
        covered = in_slot.any(dim=0)
        token_to_slot = in_slot.long().argmax(dim=0)                   # (N,)
        local_pos = (arange_n.squeeze(0) - v_starts[token_to_slot]).clamp(min=0)
        V_resid_full = torch.zeros(N, D, device=device, dtype=torch.float32)
        cov_idx = covered.nonzero(as_tuple=True)[0]
        V_resid_full[cov_idx] = V_resid[token_to_slot[cov_idx], local_pos[cov_idx]].float()
        sel_tok = sel_v[:, token_to_slot.clamp(min=0)] & covered.unsqueeze(0)  # (Q,N)
        n_v_reads = int(sel_v.sum().item())

    # ── combine ──
    # exact: one renormalized softmax carries both the K logit shift and the V
    # residual.  With no K slots read it is algebraically the linear V term, so
    # fall through to keep that case bit-identical (and READ=0 → delta stays 0).
    if mode == "exact" and wK is not None:
        delta = exact_softmax_correction(
            A=A, ds=wK / math.sqrt(D), V_base=V_base, O_base=O_base,
            RV_full=V_resid_full, sel_tok=sel_tok,
        )
    else:
        if wK is not None:
            AwK = A * wK                                               # (Q, N)
            delta_K = AwK @ V_base - AwK.sum(dim=1, keepdim=True) * O_base
            delta = delta + k_corr_scale * delta_K
        if sel_tok is not None:
            delta = delta + (A * sel_tok.float()) @ V_resid_full

    if debug_stats is not None:
        debug_stats["v_slots_read"] = debug_stats.get("v_slots_read", 0) + n_v_reads
        debug_stats["k_slots_read"] = debug_stats.get("k_slots_read", 0) + n_k_reads
    if _CHUNK_REC is not None:
        _CHUNK_REC.append(dict(
            layer_id=layer_id, kv_head=kv_head, q_offset=_q_offset, Qn=Qn, N=N,
            delta=delta.detach().clone(),
            score_V=(score_V.detach().clone() if score_V is not None else None),
            sel_v=(sel_v.detach().clone() if sel_v is not None else None),
            score_K=(score_K.detach().clone() if score_K is not None else None),
            sel_k=(sel_k.detach().clone() if sel_k is not None else None),
            budget_v=budget_v, budget_k=budget_k))
    return delta


# ─────────────────────────────────────────────
# Per-query-head decode attention
# ─────────────────────────────────────────────

class CAREKVAttention:
    """
    One decode step for a single query head.  Reads base K/V from the
    KV-head-indexed cache, computes base attention, then applies sparse
    correction via apply_slot_corrections.
    """

    def __init__(self, cfg: CacheConfig, layer_id: int, device: torch.device):
        self.cfg = cfg
        self.layer_id = layer_id
        self.device = device
        self.router = ResidualRouter(cfg, layer_id, device)
        self.base_qcfg = QuantConfig(bits=cfg.base_bits, group_size=cfg.group_size)
        self.scale_factor = 1.0 / math.sqrt(cfg.head_dim)

    def forward(
        self,
        cache: CAREKVCache,
        kv_head: int,
        q: Tensor,                  # (D,)  current query (post-RoPE)
        page_ids: Optional[List[int]] = None,
        kind: str = "both",
        k_corr_scale: float = 1.0,
        score_normalize: bool = False,
        debug_stats: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        cfg = self.cfg
        if page_ids is None:
            page_ids = cache.all_page_ids(self.layer_id, kv_head)
        if not page_ids:
            return torch.zeros(cfg.head_dim, device=self.device, dtype=q.dtype)

        # ── Dequantize valid base KV ──────────────────────────────────
        if getattr(cfg, "base_quantizer", "uniform") in {"kivi_style", "rotatekv_style", "kvquant_style"}:
            # KIVI path: K_hat/V_hat are stored fp16 in the side buffer;
            # no dequant needed.
            K_hat_concat, V_hat_concat, valid_lens = cache.read_base_hat_concat(
                self.layer_id, kv_head, page_ids,
            )
            N_valid = K_hat_concat.shape[0]
            if N_valid == 0:
                return torch.zeros(cfg.head_dim, device=self.device, dtype=q.dtype)
            K_base = K_hat_concat.float()
            V_base = V_hat_concat.float()
        else:
            Kc, Ks, Vc, Vs, valid_lens = cache.read_base_concat(
                self.layer_id, kv_head, page_ids,
            )
            N_valid = Kc.shape[0]
            if N_valid == 0:
                return torch.zeros(cfg.head_dim, device=self.device, dtype=q.dtype)
            K_base = dequantize(Kc, Ks, (N_valid, cfg.head_dim), self.base_qcfg).float()
            V_base = dequantize(Vc, Vs, (N_valid, cfg.head_dim), self.base_qcfg).float()
        q_f = q.float()

        # ── Base attention ────────────────────────────────────────────
        s_base = (K_base @ q_f) * self.scale_factor         # (N_valid,)
        a_base = F.softmax(s_base, dim=0)                   # (N_valid,)
        O_base = (a_base.unsqueeze(-1) * V_base).sum(0)     # (D,)

        # ── Correction via stored slots ───────────────────────────────
        if cfg.read_budget_ratio > 0 and kind in {"v", "k", "both"}:
            delta_O = apply_slot_corrections(
                cache=cache, cfg=cfg, router=self.router,
                layer_id=self.layer_id, kv_head=kv_head,
                page_ids=page_ids,
                q=q_f, s_base=s_base, a_base=a_base,
                V_base=V_base, O_base=O_base,
                kind=kind, k_corr_scale=k_corr_scale,
                score_normalize=score_normalize,
                debug_stats=debug_stats,
            ).float()
            O = O_base + delta_O
        else:
            O = O_base
        return O.to(q.dtype)


class CAREKVMultiHeadAttention:
    """
    Multi-head wrapper.  GQA-aware: query heads share KV heads.
    """

    def __init__(self, cfg: CacheConfig, layer_id: int, device: torch.device):
        self.cfg = cfg
        self.layer_id = layer_id
        self.device = device
        Hq = cfg.num_heads
        Hkv = cfg.num_kv_heads
        assert Hq % Hkv == 0, (Hq, Hkv)
        self.kv_group = Hq // Hkv
        self.attn = CAREKVAttention(cfg, layer_id, device)

    def forward(
        self,
        cache: CAREKVCache,
        Q: Tensor,               # (num_heads, D) — post-RoPE queries
        page_ids: Optional[List[int]] = None,
        kind: str = "both",
        k_corr_scale: float = 1.0,
        score_normalize: bool = False,
        debug_stats: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        outputs = []
        for h in range(self.cfg.num_heads):
            kv_h = h // self.kv_group
            o_h = self.attn.forward(
                cache, kv_h, Q[h], page_ids,
                kind=kind, k_corr_scale=k_corr_scale,
                score_normalize=score_normalize,
                debug_stats=debug_stats,
            )
            outputs.append(o_h)
        return torch.stack(outputs, dim=0)   # (H, D)
