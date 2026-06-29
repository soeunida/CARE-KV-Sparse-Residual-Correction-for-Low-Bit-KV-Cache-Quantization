"""
care_kv/residual_router.py
---------------------------
Decode-time / stored-prefill residual routing.

Given base-attention quantities (s_base, a_base, O_base) computed against
dequantized base K/V, plus per-page metadata recorded by the store manager,
select which stored residual slots to read.

V slot score:
    score = blk_attn_mass × v_error_norm × sensitivity

K slot score (decision-boundary-aware):
    score = page_attn_mass
          × |q_sketch · R_K_sketch|       (estimated |q·R_K| via sketch)
          × page_boundary_risk            (attn-weighted 1/(margin+ε))
          × page_v_diff                   (attn-weighted ||V_t - O_base||)
          × sensitivity

Returns the slots to actually load, capped by read_budget_ratio.
"""

from __future__ import annotations
import torch
from torch import Tensor
from typing import List, Tuple, Optional
import math
import os

from .cache import CAREKVCache, CacheConfig, PageMeta, layer_budget_multiplier
from .residual_store import get_sketch_proj


EPS = 1e-6


def _resolve_read_budgets(cfg, kind: str, total_k: int, total_v: int,
                          layer_id: int = 0) -> Tuple[int, int]:
    """Return (budget_k, budget_v) honoring cfg.read_budget_mode + per-kind
    overrides + global cfg.read_budget_ratio + per-layer policy multiplier.

    `kind` filters: budget for a disabled kind is forced to 0.

    Invariants:
      - cfg.read_budget_ratio <= 0 AND no per-kind override → (0, 0)
      - any positive resolved ratio is floored at 1 (so tiny ratios don't
        silently degenerate to zero reads via int() truncation)
      - per-layer multiplier is averaged to 1.0 across layers so total
        budget across the network is preserved.
    """
    def _resolved_ratio(override, fallback):
        if override is None or override < 0:
            return fallback
        return override

    lm = layer_budget_multiplier(cfg, layer_id)

    # adaptive_score uses the same MAX caps as absolute; the score-threshold
    # filtering happens later in route() *after* the policy ranking.
    if cfg.read_budget_mode in ("absolute", "adaptive_score"):
        bk = int(round(max(0, int(cfg.read_abs_k)) * lm))
        bv = int(round(max(0, int(cfg.read_abs_v)) * lm))
    else:
        rk = _resolved_ratio(cfg.read_budget_ratio_k, cfg.read_budget_ratio) * lm
        rv = _resolved_ratio(cfg.read_budget_ratio_v, cfg.read_budget_ratio) * lm
        bk = max(1, int(total_k * rk)) if rk > 0 and total_k > 0 else 0
        bv = max(1, int(total_v * rv)) if rv > 0 and total_v > 0 else 0

    if kind == "v":
        bk = 0
    elif kind == "k":
        bv = 0
    # cap to available
    bk = min(bk, total_k)
    bv = min(bv, total_v)
    return bk, bv


class ResidualRouter:
    """Stateless utility called once per query (per layer, per query head)."""

    def __init__(self, cfg: CacheConfig, layer_id: int, device: torch.device):
        self.cfg = cfg
        self.layer_id = layer_id
        self.device = device
        self.sensitivity = cfg.layer_sensitivity[layer_id]

    def route(
        self,
        cache: CAREKVCache,
        kv_head: int,
        page_ids: List[int],
        q: Tensor,           # (D,) current query vector
        s_base: Tensor,      # (N_valid,) base attention logits over valid tokens
        a_base: Tensor,      # (N_valid,) base attention weights (after softmax)
        O_base: Tensor,      # (D,) base attention output
        V_base_full: Tensor, # (N_valid, D) dequantized V for ||V - O_base||
        kind: str = "both",  # "v" / "k" / "both" — restricts which slot types to rank
        score_normalize: bool = False,  # if True and kind=="both", normalize
                                        # K and V scores by their per-kind mean
                                        # before merging so neither kind
                                        # dominates purely from score scale
        debug_stats: Optional[dict] = None,
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        """
        Select stored residual slots to read.

        s_base / a_base / V_base_full are aligned to **valid** tokens
        (the same length as the concatenation that read_base_concat returns).

        Returns
        -------
        k_selected : list of (page_id, channel_group_idx, slot_id)
        v_selected : list of (page_id, token_block_idx, slot_id)
        """
        cfg = self.cfg
        q = q.float()
        s_base = s_base.float()
        a_base = a_base.float()
        O_base = O_base.float()
        V_base_full = V_base_full.float()

        if a_base.numel() == 0:
            return [], []

        s_top = s_base.max().item()

        # Sketch projection for fast |q·R_K|.
        P = get_sketch_proj(cfg.head_dim, cfg.k_channel_group,
                            cfg.sketch_dim, self.device)
        num_cg = cfg.head_dim // cfg.k_channel_group
        q_sketch_by_cg = {}
        for cg in range(num_cg):
            c0 = cg * cfg.k_channel_group
            c1 = c0 + cfg.k_channel_group
            q_sketch_by_cg[cg] = q[c0:c1] @ P                   # (sketch_dim,)

        # Walk pages, accumulating per-page token offsets from valid_tokens.
        k_candidates: List[Tuple[float, int, int, int]] = []   # (score, pid, cg, slot)
        v_candidates: List[Tuple[float, int, int, int]] = []   # (score, pid, vb, slot)

        token_offset = 0
        for page_id in page_ids:
            meta: Optional[PageMeta] = cache.meta_table[self.layer_id][kv_head][page_id]
            n_valid = int(cache.valid_tokens[self.layer_id, kv_head, page_id].item())
            if n_valid == 0 or meta is None:
                continue

            t0, t1 = token_offset, token_offset + n_valid
            page_a = a_base[t0:t1]
            page_s = s_base[t0:t1]
            page_V = V_base_full[t0:t1]
            page_attn_mass = page_a.sum().item()

            # Page-level boundary risk and V-diff (shared across CGs of this page).
            margins = (s_top - page_s).clamp(min=EPS)
            page_boundary_risk = (page_a * (1.0 / margins)).sum().item()
            V_diff_norms = (page_V - O_base.unsqueeze(0)).norm(dim=-1)
            page_v_diff = (page_a * V_diff_norms).sum().item()

            # Pick the score formula for this run (ablation hook). For the
            # paper-best path (baseline_score="carekv") this is byte-identical
            # to the original formula.
            baseline = getattr(cfg, "baseline_score", "carekv")

            # ── K residual candidates (only stored slots) ─────────────
            if kind in {"k", "both"} and meta.k_sketch is not None:
                for cg, slot in enumerate(meta.k_residual_slots):
                    if slot < 0:
                        continue
                    rk_sketch = meta.k_sketch[cg].float().to(q.device)
                    qdotr_est = (q_sketch_by_cg[cg] * rk_sketch).sum().abs().item()
                    if baseline == "carekv":
                        score = (
                            page_attn_mass
                            * qdotr_est
                            * page_boundary_risk
                            * page_v_diff
                            * self.sensitivity
                        )
                    elif baseline == "random":
                        score = float(torch.rand(1).item())
                    elif baseline == "magnitude_only":
                        score = qdotr_est
                    elif baseline == "attention_only":
                        score = page_attn_mass
                    elif baseline == "oracle_proxy":
                        # Magnitude × attention only — drops structural-prior
                        # and sensitivity multipliers; diagnostic upper bound.
                        score = qdotr_est * page_attn_mass
                    else:
                        raise ValueError(f"unknown baseline_score: {baseline}")
                    k_candidates.append((score, page_id, cg, slot))

            # ── V residual candidates (only stored slots) ─────────────
            if kind not in {"v", "both"}:
                # Skip V scoring entirely for kind=="k"
                token_offset = t1
                continue
            for vb, slot in enumerate(meta.v_residual_slots):
                if slot < 0:
                    continue
                t_vb0 = vb * cfg.v_token_block
                t_vb1 = min(t_vb0 + cfg.v_token_block, n_valid)
                if t_vb1 <= t_vb0:
                    continue
                blk_a = page_a[t_vb0:t_vb1]
                blk_attn_mass = blk_a.sum().item()

                v_err = (
                    meta.v_error_norm[vb].item()
                    if meta.v_error_norm is not None else 1.0
                )
                if baseline == "carekv":
                    score = blk_attn_mass * v_err * self.sensitivity
                elif baseline == "random":
                    score = float(torch.rand(1).item())
                elif baseline == "magnitude_only":
                    score = v_err
                elif baseline == "attention_only":
                    score = blk_attn_mass
                elif baseline == "oracle_proxy":
                    score = blk_attn_mass * v_err
                else:
                    raise ValueError(f"unknown baseline_score: {baseline}")
                # Phase 11B (EXPERIMENTAL, gated): position-aware read re-weight of V
                # slots. Default CAREKV_POSITION_POLICY=none → no-op (byte-identical).
                _ppol = os.environ.get("CAREKV_POSITION_POLICY", "none")
                if _ppol != "none":
                    from .residual_store import _position_weight as _pw
                    _sl = int(os.environ.get("CAREKV_SEQ_LEN", "0") or 0)
                    score = score * _pw(_ppol, token_offset + t_vb0, _sl)
                v_candidates.append((score, page_id, vb, slot))

            token_offset = t1

        # ── Apply read budget over all stored slots ───────────────────
        total_k = len(k_candidates)
        total_v = len(v_candidates)
        if total_k + total_v == 0:
            return [], []

        # Resolve per-kind read budgets honoring (a) absolute vs ratio mode,
        # (b) per-kind overrides, (c) per-layer policy multiplier (Phase E).
        budget_k, budget_v = _resolve_read_budgets(
            cfg, kind, total_k, total_v, layer_id=self.layer_id,
        )

        # Preserve R=0 ≡ base_quant invariant: if BOTH kinds resolved to 0
        # budget AND the global ratio is also 0 (or kind is single-mode and
        # its budget is 0), we return empty.
        if budget_k == 0 and budget_v == 0:
            return [], []

        # Optional per-kind score normalization (kind="both" only).
        def _norm(cands):
            if not cands:
                return cands
            mean = sum(abs(c[0]) for c in cands) / len(cands)
            scale = mean if mean > EPS else EPS
            return [(c[0] / scale, *c[1:]) for c in cands]

        if score_normalize and kind == "both":
            k_for_rank = _norm(k_candidates)
            v_for_rank = _norm(v_candidates)
        else:
            k_for_rank = k_candidates
            v_for_rank = v_candidates

        policy = getattr(cfg, "route_policy", "separate")

        if policy == "joint":
            # Combined ranking (legacy) — budgets are merged as a global
            # cap to preserve the previous behavior; kind-specific budgets
            # if both > 0 are summed.
            merged_budget = budget_k + budget_v if (budget_k > 0 and budget_v > 0) \
                else max(budget_k, budget_v)
            all_cands = (
                [("K", *c) for c in k_for_rank] +
                [("V", *c) for c in v_for_rank]
            )
            all_cands.sort(key=lambda x: x[1], reverse=True)
            selected = all_cands[:merged_budget]

        elif policy == "k_first":
            # Fill K budget first (top-budget_k from K candidates), then
            # spend the remaining (budget_k + budget_v - len(k_picked))
            # global allowance on V candidates whose score exceeds a small
            # quantile threshold (mean of V scores).
            k_sorted = sorted(k_for_rank, key=lambda c: c[0], reverse=True)
            k_pick = k_sorted[:budget_k]
            remaining = max(0, budget_k + budget_v - len(k_pick))
            v_sorted = sorted(v_for_rank, key=lambda c: c[0], reverse=True)
            if v_sorted and remaining > 0:
                vmean = sum(c[0] for c in v_sorted) / len(v_sorted)
                v_pick = [c for c in v_sorted if c[0] >= vmean][:remaining]
            else:
                v_pick = []
            selected = ([("K", *c) for c in k_pick]
                        + [("V", *c) for c in v_pick])

        elif policy == "adaptive":
            # Allocate the (budget_k + budget_v) total based on attention
            # entropy.  High entropy → more V (averaging dominates);
            # low entropy → more K (boundary risk dominates).
            total_budget = budget_k + budget_v
            if total_budget <= 0:
                selected = []
            else:
                a = a_base.clamp(min=EPS)
                ent = float(-(a * a.log()).sum().item())
                max_ent = math.log(max(a_base.numel(), 2))
                v_frac = max(0.0, min(1.0, ent / max_ent))    # [0, 1]
                v_alloc = max(0, int(round(total_budget * v_frac)))
                k_alloc = max(0, total_budget - v_alloc)
                k_sorted = sorted(k_for_rank, key=lambda c: c[0], reverse=True)[:k_alloc]
                v_sorted = sorted(v_for_rank, key=lambda c: c[0], reverse=True)[:v_alloc]
                selected = ([("K", *c) for c in k_sorted]
                            + [("V", *c) for c in v_sorted])

        else:   # "separate" (default, paper-ready)
            # Pick top-budget_k K and top-budget_v V independently — no
            # cross-kind score comparison.  Each kind operates in its own
            # ranking pool with its own budget.
            k_sorted = sorted(k_for_rank, key=lambda c: c[0], reverse=True)[:budget_k]
            v_sorted = sorted(v_for_rank, key=lambda c: c[0], reverse=True)[:budget_v]
            selected = ([("K", *c) for c in k_sorted]
                        + [("V", *c) for c in v_sorted])

        # ── Adaptive read-budget thresholding ─────────────────────
        # When read_budget_mode == "adaptive_score", the per-kind read
        # budgets above act as MAX caps and we additionally drop slots
        # whose score is below threshold(s). Default (fixed) mode keeps
        # the selected list unchanged — preserves paper-best behavior.
        skipped_k_rel = skipped_k_abs = skipped_v_rel = skipped_v_abs = 0
        if getattr(cfg, "read_budget_mode", "ratio") == "adaptive_score":
            rel_thr = float(getattr(cfg, "read_relative_threshold", 0.0) or 0.0)
            abs_thr = float(getattr(cfg, "read_absolute_threshold", 0.0) or 0.0)
            min_keep = int(getattr(cfg, "read_min_keep", 0) or 0)

            sel_k = [c for c in selected if c[0] == "K"]
            sel_v = [c for c in selected if c[0] == "V"]
            # selected has scores at c[1] (we re-sort defensively in case
            # the joint dispatch interleaved K and V).
            sel_k.sort(key=lambda c: c[1], reverse=True)
            sel_v.sort(key=lambda c: c[1], reverse=True)

            def _filter(lst):
                if not lst:
                    return lst, 0, 0
                top = float(abs(lst[0][1]))
                gate = max(rel_thr * top, abs_thr)
                kept = []
                drop_rel = drop_abs = 0
                for c in lst:
                    s = float(abs(c[1]))
                    pass_rel = (rel_thr <= 0) or (s >= rel_thr * top)
                    pass_abs = (abs_thr <= 0) or (s >= abs_thr)
                    if pass_rel and pass_abs:
                        kept.append(c)
                    else:
                        if not pass_rel:
                            drop_rel += 1
                        if not pass_abs:
                            drop_abs += 1
                # Enforce min_keep floor (if any candidates existed)
                if min_keep > 0 and len(kept) < min_keep and len(lst) > 0:
                    kept = lst[: min(min_keep, len(lst))]
                return kept, drop_rel, drop_abs

            sel_k, skipped_k_rel, skipped_k_abs = _filter(sel_k)
            sel_v, skipped_v_rel, skipped_v_abs = _filter(sel_v)
            selected = sel_k + sel_v

        k_selected = [(c[2], c[3], c[4]) for c in selected if c[0] == "K"]
        v_selected = [(c[2], c[3], c[4]) for c in selected if c[0] == "V"]

        # Per-route debug breakdown (accumulated by caller).
        if debug_stats is not None:
            def _stats(cands):
                if not cands:
                    return 0.0, 0.0
                vals = [c[0] for c in cands]
                return (sum(abs(v) for v in vals) / len(vals),
                        max(abs(v) for v in vals))
            k_mean, k_max = _stats(k_candidates)
            v_mean, v_max = _stats(v_candidates)
            kn_mean, kn_max = _stats(k_for_rank)
            vn_mean, vn_max = _stats(v_for_rank)
            for k, v in [
                ("router_n_k_cands", len(k_candidates)),
                ("router_n_v_cands", len(v_candidates)),
                ("router_k_score_mean_sum", k_mean),
                ("router_k_score_max_sum", k_max),
                ("router_v_score_mean_sum", v_mean),
                ("router_v_score_max_sum", v_max),
                ("router_k_score_norm_mean_sum", kn_mean),
                ("router_v_score_norm_mean_sum", vn_mean),
                ("router_n_k_selected", len(k_selected)),
                ("router_n_v_selected", len(v_selected)),
                ("router_score_normalize", int(bool(score_normalize and kind == "both"))),
                # Adaptive-mode counters (zero when not in adaptive_score mode)
                ("router_requested_RK", budget_k),
                ("router_requested_RV", budget_v),
                ("router_effective_RK_sum", len(k_selected)),
                ("router_effective_RV_sum", len(v_selected)),
                ("router_skipped_K_by_relative_threshold", skipped_k_rel),
                ("router_skipped_K_by_absolute_threshold", skipped_k_abs),
                ("router_skipped_V_by_relative_threshold", skipped_v_rel),
                ("router_skipped_V_by_absolute_threshold", skipped_v_abs),
                ("router_n_route_calls", 1),
            ]:
                debug_stats[k] = debug_stats.get(k, 0) + v

        return k_selected, v_selected
