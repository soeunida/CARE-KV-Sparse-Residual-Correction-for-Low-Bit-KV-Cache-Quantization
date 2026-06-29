"""tools/make_carekv_before_after_3d.py

CARE-KV vs base_quant K/V error diagnostics — visualization focused on
making the sparse residual restoration unambiguously visible.

Plot modes (`--plot-mode`):

  error-decomposition   (DEFAULT, 4 panels per layer per kind)
    1) |X_fp − X_hat|            base_quant error
    2) |X_fp − X_care|           CARE-KV remaining error  (shared vmin/vmax with 1)
    3) (1) − (2)                 signed error reduction (diverging RdBu_r)
    4) |X_care − X_hat|          recovered sparse residual mass
    Outputs:
      3d_carekv_{K,V}_error_layer{NN}.png            (3D surface)
      heatmap_carekv_{K,V}_error_layer{NN}.png       (2D imshow — primary)

  visible-error         (5 panels per layer per kind)
    1) |X_fp|                    original magnitude (optional scatter overlay)
    2) base error
    3) CARE-KV remaining error
    4) signed error reduction (diverging RdBu_r)
    5) recovered sparse residual
    Outputs:
      3d_carekv_{K,V}_visible_layer{NN}.png          (3D surface)
      heatmap_carekv_{K,V}_visible_layer{NN}.png     (2D imshow)

  clean-error           (3-panel 3D + 4-panel 2D heatmap)
    3D — 3 panels:
      1) base error (plot_surface)
      2) CARE-KV remaining error (plot_surface, shared vmin/vmax with 1)
      3) recovered residual as scatter3D, top --overlay-top-percent cells only
    2D — same 4-panel heatmap as error-decomposition (this is the PRIMARY
         paper figure per the latest spec).
    Outputs:
      3d_carekv_{K,V}_error_clean_layer{NN}.png       (3D, clean/sparse)
      heatmap_carekv_{K,V}_error_layer{NN}.png        (2D, same as above)

  both                  produces error-decomposition + visible-error
  all                   produces error-decomposition + visible-error + clean-error

Color design (current revision):
  --error-cmap     default "inferno"   sequential, used for panels 1+2
                                       (vmin=0, vmax=percentile(BASE_error, p))
                                       same scale on both panels so care
                                       error visibly looks dimmer than base
  --reduction-cmap default "RdBu_r"    diverging, used for panel 3
                                       TwoSlopeNorm(vcenter=0),
                                       vlim = percentile(|reduction|, p)
                                       positive=red, zero=white, negative=blue
                                       title: "Signed error reduction (positive = improved)"
  --residual-cmap  default "hot"       sequential, used for panel 4 / scatter
                                       vmin=0, vmax=percentile(recovered, p)
  --error-percentile default 99
  --error-gain      default 1.0  (visualization only, applied to error panels)
  --log-error       default off  (plot log1p(error*gain) when set)
  --overlay-recovered-mask         visible-error: scatter top recovered cells over |X_fp|
  --overlay-top-percent default 1.0 (cells above this percentile of recovered magnitude
                                     are scattered; also used by clean-error 3D panel 3)

Store-time selection matches paper-best:
  page_size=16, k_channel_group=32, v_token_block implicitly via STORE_ABS_V,
  STORE_ABS_K=2, STORE_ABS_V=4, score = L2 norm of residual segment.

Usage:
  PYTHONPATH=/home/soeun python tools/make_carekv_before_after_3d.py \
      --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --out-dir results/paper_eval_20260529_015053/figures \
      --layers 0 11 21 --seq-len 512 \
      --max-tokens 128 --max-channels 128 \
      --plot-mode clean-error
"""
from __future__ import annotations
import argparse, json, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, PowerNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from CARE_KV.care_kv.quantizer import QuantConfig, quantize_and_residual


# ─────────────────────────────────────────────
# Store-time selection (paper-best knobs)
# ─────────────────────────────────────────────

def _select_k_stored_mask(R_K, page_size, k_channel_group, store_abs_k):
    Hkv, T, D = R_K.shape
    assert D % k_channel_group == 0
    G = D // k_channel_group
    mask = torch.zeros_like(R_K, dtype=torch.bool)
    for p_start in range(0, T, page_size):
        p_end = min(p_start + page_size, T)
        for h in range(Hkv):
            for g in range(G):
                c0, c1 = g * k_channel_group, (g + 1) * k_channel_group
                segs = R_K[h, p_start:p_end, c0:c1]
                scores = segs.float().norm(dim=-1)
                k = min(store_abs_k, segs.shape[0])
                top_idx = torch.topk(scores, k).indices
                mask[h, p_start + top_idx, c0:c1] = True
    return mask


def _select_v_stored_mask(R_V, page_size, store_abs_v):
    Hkv, T, D = R_V.shape
    mask = torch.zeros_like(R_V, dtype=torch.bool)
    for p_start in range(0, T, page_size):
        p_end = min(p_start + page_size, T)
        page = R_V[:, p_start:p_end, :]
        scores = page.float().norm(dim=-1)
        for h in range(Hkv):
            k = min(store_abs_v, page.shape[1])
            top_idx = torch.topk(scores[h], k).indices
            mask[h, p_start + top_idx, :] = True
    return mask


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _stride_downsample(arr, max_dim, axis):
    n = arr.shape[axis]
    if n <= max_dim:
        return arr
    step = max(1, n // max_dim)
    idx = np.arange(0, n, step)[:max_dim]
    return np.take(arr, idx, axis=axis)


def _apply_view_xform(arr, error_gain, log_error):
    a = arr * error_gain
    if log_error:
        a = np.log1p(np.clip(a, 0, None))
    return a


def _percentile(arr, p, default=1.0):
    if arr.size == 0:
        return default
    v = float(np.percentile(arr, p))
    return v if v > 0 else default


def _reduction_palette(err_reduction, reduction_cmap, error_percentile):
    """Always-diverging palette centered at 0."""
    span = _percentile(np.abs(err_reduction), error_percentile, default=1.0)
    # vmin/vmax sanity for TwoSlopeNorm
    span = max(span, 1e-12)
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)
    return (reduction_cmap, norm, -span, span,
            "Signed error reduction (positive = improved)")


def _suffix(error_gain, log_error):
    s = []
    if error_gain != 1.0: s.append(f"×{error_gain:g}")
    if log_error:         s.append("log1p")
    return f"  ({', '.join(s)})" if s else ""


# ─────────────────────────────────────────────
# 3D surface primitive
# ─────────────────────────────────────────────

def _surface(ax, arr, title, vmin, vmax, cmap,
             max_tokens, max_channels, zlabel, norm=None):
    ds = _stride_downsample(arr, max_tokens, axis=0)
    ds = _stride_downsample(ds, max_channels, axis=1)
    T, C = ds.shape
    X, Y = np.meshgrid(np.arange(C), np.arange(T))
    rstride = max(1, T // 64)
    cstride = max(1, C // 64)
    if norm is not None:
        surf = ax.plot_surface(X, Y, ds, cmap=cmap, norm=norm,
                                linewidth=0, antialiased=False,
                                rstride=rstride, cstride=cstride)
        ax.set_zlim(norm.vmin, norm.vmax)
    else:
        surf = ax.plot_surface(X, Y, ds, cmap=cmap, vmin=vmin, vmax=vmax,
                                linewidth=0, antialiased=False,
                                rstride=rstride, cstride=cstride)
        ax.set_zlim(vmin, vmax)
    ax.set_xlabel("Channel"); ax.set_ylabel("Token"); ax.set_zlabel(zlabel)
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=28, azim=-58)
    return surf


def _overlay_recovered_scatter(ax, X_fp_abs, recovered,
                                max_tokens, max_channels, top_percent):
    X_ds = _stride_downsample(_stride_downsample(X_fp_abs, max_tokens, 0),
                              max_channels, 1)
    rec_ds = _stride_downsample(_stride_downsample(recovered, max_tokens, 0),
                                max_channels, 1)
    nonzero = rec_ds[rec_ds > 0]
    if nonzero.size == 0:
        return
    thresh = float(np.percentile(nonzero, 100.0 - top_percent))
    sel = rec_ds >= thresh
    ys, xs = np.where(sel)
    if ys.size == 0:
        return
    zs = X_ds[ys, xs]
    ax.scatter(xs, ys, zs, c="red", s=4, alpha=0.8, depthshade=False,
               label=f"top {top_percent:.1f}% recovered")
    ax.legend(loc="upper right", fontsize=7)


# ─────────────────────────────────────────────
# 3D plotters
# ─────────────────────────────────────────────

def _plot_3d_error_4panel(
    err_base, err_care, err_reduction, recovered,
    layer_id, kind, out_path,
    max_tokens, max_channels,
    error_cmap, reduction_cmap, residual_cmap,
    error_percentile, error_gain, log_error,
):
    eb = _apply_view_xform(err_base, error_gain, log_error)
    ec = _apply_view_xform(err_care, error_gain, log_error)
    err_vmax = _percentile(eb, error_percentile)

    red_cmap, red_norm, red_vmin, red_vmax, red_title = \
        _reduction_palette(err_reduction, reduction_cmap, error_percentile)
    rec_vmax = _percentile(recovered, error_percentile)
    suf = _suffix(error_gain, log_error)

    fig = plt.figure(figsize=(20, 6))
    panels = [
        (1, f"|{kind}_fp − {kind}_hat|        (base_quant){suf}",
         eb, 0.0, err_vmax, error_cmap, None, "magnitude"),
        (2, f"|{kind}_fp − {kind}_care|       (CARE-KV){suf}",
         ec, 0.0, err_vmax, error_cmap, None, "magnitude"),
        (3, red_title,
         err_reduction, red_vmin, red_vmax, red_cmap, red_norm, "Δ"),
        (4, f"Recovered |{kind}_care − {kind}_hat|",
         recovered, 0.0, rec_vmax, residual_cmap, None, "magnitude"),
    ]
    for idx, title, arr, vmin, vmax, cmap, norm, zlab in panels:
        ax = fig.add_subplot(1, 4, idx, projection="3d")
        surf = _surface(ax, arr, title, vmin, vmax, cmap,
                        max_tokens, max_channels, zlab, norm=norm)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    fig.suptitle(f"CARE-KV vs base_quant — {kind} error decomposition, layer {layer_id}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_3d_visible_5panel(
    X_fp_abs, err_base, err_care, err_reduction, recovered,
    layer_id, kind, out_path,
    max_tokens, max_channels,
    error_cmap, reduction_cmap, residual_cmap,
    error_percentile, error_gain, log_error,
    overlay, overlay_top_percent,
):
    eb = _apply_view_xform(err_base, error_gain, log_error)
    ec = _apply_view_xform(err_care, error_gain, log_error)
    err_vmax = _percentile(eb, error_percentile)
    red_cmap, red_norm, red_vmin, red_vmax, red_title = \
        _reduction_palette(err_reduction, reduction_cmap, error_percentile)
    rec_vmax = _percentile(recovered, error_percentile)
    fp_vmax  = _percentile(X_fp_abs, error_percentile)
    suf = _suffix(error_gain, log_error)

    fig = plt.figure(figsize=(26, 6))
    ax1 = fig.add_subplot(1, 5, 1, projection="3d")
    surf1 = _surface(ax1, X_fp_abs, f"Original {kind} magnitude  |{kind}_fp|",
                     0.0, fp_vmax, "viridis", max_tokens, max_channels, "|value|")
    fig.colorbar(surf1, ax=ax1, shrink=0.55, pad=0.08)
    if overlay:
        _overlay_recovered_scatter(ax1, X_fp_abs, recovered,
                                    max_tokens, max_channels, overlay_top_percent)

    panels = [
        (2, f"|{kind}_fp − {kind}_hat|        (base_quant){suf}",
         eb, 0.0, err_vmax, error_cmap, None, "magnitude"),
        (3, f"|{kind}_fp − {kind}_care|       (CARE-KV){suf}",
         ec, 0.0, err_vmax, error_cmap, None, "magnitude"),
        (4, red_title,
         err_reduction, red_vmin, red_vmax, red_cmap, red_norm, "Δ"),
        (5, f"Recovered |{kind}_care − {kind}_hat|",
         recovered, 0.0, rec_vmax, residual_cmap, None, "magnitude"),
    ]
    for idx, title, arr, vmin, vmax, cmap, norm, zlab in panels:
        ax = fig.add_subplot(1, 5, idx, projection="3d")
        surf = _surface(ax, arr, title, vmin, vmax, cmap,
                        max_tokens, max_channels, zlab, norm=norm)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    fig.suptitle(f"CARE-KV residual restoration (visible) — {kind}, layer {layer_id}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _plot_3d_clean_3panel(
    err_base, err_care, recovered,
    layer_id, kind, out_path,
    max_tokens, max_channels,
    error_cmap, residual_cmap,
    error_percentile, error_gain, log_error,
    overlay_top_percent,
):
    """Cleaner 3D: 2 plot_surface panels for error + 1 scatter3D for recovered."""
    eb = _apply_view_xform(err_base, error_gain, log_error)
    ec = _apply_view_xform(err_care, error_gain, log_error)
    err_vmax = _percentile(eb, error_percentile)
    rec_vmax_full = _percentile(recovered, error_percentile)
    suf = _suffix(error_gain, log_error)

    fig = plt.figure(figsize=(18, 6))
    for idx, title, arr in [
        (1, f"|{kind}_fp − {kind}_hat|        (base_quant){suf}", eb),
        (2, f"|{kind}_fp − {kind}_care|       (CARE-KV){suf}",    ec),
    ]:
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        surf = _surface(ax, arr, title, 0.0, err_vmax, error_cmap,
                        max_tokens, max_channels, "magnitude")
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)

    # Panel 3: scatter3D for top-X% recovered cells (full resolution)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    nonzero = recovered[recovered > 0]
    if nonzero.size > 0:
        thresh = float(np.percentile(nonzero, 100.0 - overlay_top_percent))
        ys, xs = np.where(recovered >= thresh)
        zs = recovered[ys, xs]
        if zs.size > 0:
            sc = ax3.scatter(xs, ys, zs, c=zs, cmap=residual_cmap,
                             s=6, alpha=0.85, depthshade=False,
                             vmin=0, vmax=rec_vmax_full)
            fig.colorbar(sc, ax=ax3, shrink=0.55, pad=0.08)
            ax3.set_xlim(0, recovered.shape[1])
            ax3.set_ylim(0, recovered.shape[0])
            ax3.set_zlim(0, rec_vmax_full)
    ax3.set_xlabel("Channel"); ax3.set_ylabel("Token"); ax3.set_zlabel("magnitude")
    ax3.set_title(f"Recovered |{kind}_care − {kind}_hat|  (top {overlay_top_percent:.1f}% scatter)",
                  fontsize=10)
    ax3.view_init(elev=28, azim=-58)

    fig.suptitle(f"CARE-KV vs base_quant — {kind} (clean), layer {layer_id}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────
# 2D heatmap plotters
# ─────────────────────────────────────────────

def _imshow_panel(ax, arr, title, vmin, vmax, cmap, norm=None):
    if norm is not None:
        im = ax.imshow(arr, cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")
    else:
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")
    ax.set_xlabel("Channel"); ax.set_ylabel("Token")
    ax.set_title(title, fontsize=10)
    return im


def _plot_2d_heatmap_5panel_primary(
    err_base, err_care, err_reduction, recovered, mask_bool,
    layer_id, kind, out_path, stats,
    error_cmap, positive_reduction_cmap, residual_cmap,
    error_percentile, error_gain, log_error,
    binary_color="#b30000",
):
    """Primary paper figure (publication polish).

    5 panels:
      1) base error |X_fp − X_hat|
      2) CARE-KV remaining error |X_fp − X_care|       (shared vmin/vmax w/ 1)
      3) positive-only error reduction max(base − care, 0) (Reds family)
         + contour overlay of binary mask boundary
      4) recovered residual magnitude |X_care − X_hat|
      5) recovered residual binary mask (white background, dark-red ink)

    A stats annotation appears under the figure title:
      "mean error: 0.0468 → 0.0323, reduction 31.1%, recovered 25.0%"
    """
    eb = _apply_view_xform(err_base, error_gain, log_error)
    ec = _apply_view_xform(err_care, error_gain, log_error)
    err_vmax = _percentile(eb, error_percentile)

    pos_red = np.clip(err_reduction, 0, None)
    pos_vmax = _percentile(pos_red[pos_red > 0]
                           if pos_red.any() else pos_red,
                           error_percentile, default=1.0)
    rec_vmax = _percentile(recovered, error_percentile)
    suf = _suffix(error_gain, log_error)

    binary_cmap = ListedColormap(["white", binary_color])

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    panels = [
        (axes[0], f"|{kind}_fp − {kind}_hat|        (base_quant){suf}",
         eb, 0.0, err_vmax, error_cmap, None),
        (axes[1], f"|{kind}_fp − {kind}_care|       (CARE-KV){suf}",
         ec, 0.0, err_vmax, error_cmap, None),
        (axes[2], "Positive error reduction by CARE-KV",
         pos_red, 0.0, pos_vmax, positive_reduction_cmap, None),
        (axes[3], f"Recovered |{kind}_care − {kind}_hat|",
         recovered, 0.0, rec_vmax, residual_cmap, None),
        (axes[4], "Recovered residual mask",
         mask_bool.astype(float), 0.0, 1.0, binary_cmap, None),
    ]
    for ax, title, arr, vmin, vmax, cmap, norm in panels:
        im = _imshow_panel(ax, arr, title, vmin, vmax, cmap, norm=norm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Contour overlay on panel 3: outline the recovered-residual mask boundary
    if mask_bool.any():
        axes[2].contour(mask_bool.astype(float), levels=[0.5],
                        colors="black", linewidths=0.35, alpha=0.7)

    # Stats annotation under the title
    base_mean = stats["base_mean"]
    care_mean = stats["care_mean"]
    rel_redux = stats["relative_error_redux"] * 100.0
    sparsity  = stats["recovered_sparsity"] * 100.0
    annot = (f"mean error: {base_mean:.4f} → {care_mean:.4f}   "
             f"reduction {rel_redux:.1f}%   recovered {sparsity:.1f}%")

    fig.suptitle(f"CARE-KV vs base_quant — {kind} error decomposition, "
                 f"layer {layer_id}", fontsize=13)
    fig.text(0.5, 0.93, annot, ha="center", fontsize=10, color="#444444")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# Backward-compatibility shim: the 4-panel function is the public name used
# by the dispatch loop. Route it to the new 5-panel primary plotter, but
# require the extra `mask_bool` + `stats` arguments at the call site.
def _plot_2d_heatmap_primary(
    err_base, err_care, err_reduction, recovered, mask_bool,
    layer_id, kind, out_path, stats,
    error_cmap, reduction_cmap, residual_cmap,
    error_percentile, error_gain, log_error,
):
    # Pick a sequential positive-only cmap for panel 3; reuse the user's
    # reduction-cmap choice if it's a known sequential palette, else fall
    # back to "Reds" (per latest spec).
    POS_CHOICES = {"Reds", "YlOrRd", "OrRd", "Oranges", "YlGnBu"}
    pos_cmap = reduction_cmap if reduction_cmap in POS_CHOICES else "Reds"
    _plot_2d_heatmap_5panel_primary(
        err_base, err_care, err_reduction, recovered, mask_bool,
        layer_id, kind, out_path, stats,
        error_cmap, pos_cmap, residual_cmap,
        error_percentile, error_gain, log_error,
    )


def _plot_2d_heatmap_5panel_visible(
    X_fp_abs, err_base, err_care, err_reduction, recovered, mask,
    layer_id, kind, out_path,
    error_cmap, reduction_cmap, residual_cmap,
    error_percentile, error_gain, log_error,
    overlay, overlay_top_percent,
):
    eb = _apply_view_xform(err_base, error_gain, log_error)
    ec = _apply_view_xform(err_care, error_gain, log_error)
    err_vmax = _percentile(eb, error_percentile)
    red_cmap, red_norm, red_vmin, red_vmax, red_title = \
        _reduction_palette(err_reduction, reduction_cmap, error_percentile)
    rec_vmax = _percentile(recovered, error_percentile)
    fp_vmax  = _percentile(X_fp_abs, error_percentile)
    suf = _suffix(error_gain, log_error)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    cells = [
        (axes[0], f"Original |{kind}_fp|",
         X_fp_abs, 0.0, fp_vmax, "viridis", None),
        (axes[1], f"|{kind}_fp − {kind}_hat|{suf}",
         eb, 0.0, err_vmax, error_cmap, None),
        (axes[2], f"|{kind}_fp − {kind}_care|{suf}",
         ec, 0.0, err_vmax, error_cmap, None),
        (axes[3], red_title,
         err_reduction, red_vmin, red_vmax, red_cmap, red_norm),
        (axes[4], f"Recovered |{kind}_care − {kind}_hat|",
         recovered, 0.0, rec_vmax, residual_cmap, None),
    ]
    for ax, title, arr, vmin, vmax, cmap, norm in cells:
        im = _imshow_panel(ax, arr, title, vmin, vmax, cmap, norm=norm)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if overlay:
        nonzero = recovered[recovered > 0]
        if nonzero.size > 0:
            thresh = float(np.percentile(nonzero, 100.0 - overlay_top_percent))
            ys, xs = np.where(recovered >= thresh)
            if ys.size:
                axes[0].scatter(xs, ys, c="red", s=2, alpha=0.6,
                                label=f"top {overlay_top_percent:.1f}% recovered")
                axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle(f"CARE-KV residual restoration (visible) — {kind}, layer {layer_id}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────
# Publication-clean 2D heatmaps (paper-clean mode)
# ─────────────────────────────────────────────

def _paper_panel(ax, arr, title, vmin, vmax, cmap):
    """Single imshow panel with paper-grade typography."""
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=10)
    return im


def _paper_axes_strip(fig, height_ratio=0.14):
    """Reserve a top "title strip" axes + a 3-column subplot row below it.

    Returns (title_ax, [ax0, ax1, ax2]).
    The title_ax is set off (no spines/ticks) so it works as a text container
    while constrained_layout reserves its full vertical share.
    """
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[height_ratio, 1.0 - height_ratio],
        hspace=0.02,
    )
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
    return title_ax, axes


def _paper_high_recovery_mask(recovered, top_percent):
    """Boolean mask of cells whose `recovered` magnitude is in the top-X%."""
    nonzero = recovered[recovered > 0]
    if nonzero.size == 0 or top_percent <= 0:
        return np.zeros_like(recovered, dtype=bool)
    thresh = float(np.percentile(nonzero, 100.0 - top_percent))
    return recovered >= thresh


def _plot_2d_paper_clean_3panel(
    err_base, err_care, err_reduction, recovered, mask_bool,
    layer_id, kind, out_path, stats,
    error_cmap, positive_reduction_cmap,
    error_percentile, reduction_percentile, reduction_gamma,
    mask_contour_top_percent,
):
    """Compact 3-panel publication figure for one (layer, kind):

      Base INT3 error  |  After CARE-KV  |  Error reduced
        ↑ shared colorbar (same scale)      ↑ separate colorbar (PowerNorm)
                                            ↑ top-X% recovered contour overlay
    """
    err_vmax = _percentile(err_base, error_percentile)

    pos_red = np.clip(err_reduction, 0, None)
    pos_nonzero = pos_red[pos_red > 0]
    pos_vmax = _percentile(pos_nonzero, reduction_percentile, default=1.0) \
               if pos_nonzero.size > 0 else 1.0
    # Gamma-stretched colormap norm (boosts small positive reductions)
    if reduction_gamma is not None and reduction_gamma > 0 and reduction_gamma != 1.0:
        pos_norm = PowerNorm(gamma=reduction_gamma, vmin=0.0, vmax=pos_vmax)
    else:
        pos_norm = None  # imshow uses vmin/vmax directly

    fig = plt.figure(figsize=(15, 5.6), layout="constrained")
    title_ax, axes = _paper_axes_strip(fig, height_ratio=0.16)

    im0 = axes[0].imshow(err_base, cmap=error_cmap,
                          vmin=0.0, vmax=err_vmax,
                          aspect="auto", interpolation="nearest")
    axes[0].set_title("Base INT3 error", fontsize=13)
    im1 = axes[1].imshow(err_care, cmap=error_cmap,
                          vmin=0.0, vmax=err_vmax,
                          aspect="auto", interpolation="nearest")
    axes[1].set_title("After CARE-KV", fontsize=13)
    if pos_norm is not None:
        im2 = axes[2].imshow(pos_red, cmap=positive_reduction_cmap,
                              norm=pos_norm,
                              aspect="auto", interpolation="nearest")
    else:
        im2 = axes[2].imshow(pos_red, cmap=positive_reduction_cmap,
                              vmin=0.0, vmax=pos_vmax,
                              aspect="auto", interpolation="nearest")
    axes[2].set_title("Error reduced", fontsize=13)

    # Subtle dark-red contour over top-X% recovered locations only
    if mask_contour_top_percent > 0:
        hi_mask = _paper_high_recovery_mask(recovered, mask_contour_top_percent)
        if hi_mask.any():
            axes[2].contour(hi_mask.astype(float), levels=[0.5],
                            colors="#8B0000", linewidths=0.35, alpha=0.4)

    # Tiny inset label on panel 3 to disambiguate sign convention
    axes[2].text(
        0.02, 0.98, "positive = improved",
        transform=axes[2].transAxes,
        fontsize=8, color="#333333", alpha=0.8,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.6),
    )

    # Axis label discipline: Token only on panel 0; Channel on all.
    for ax in axes:
        ax.set_xlabel("Channel", fontsize=11)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel("Token", fontsize=11)
    for ax in axes[1:]:
        ax.set_ylabel("")
        ax.tick_params(labelleft=True, labelsize=9)  # keep ticks, hide label

    # Shared colorbar for panels 0+1
    cb12 = fig.colorbar(im1, ax=axes[:2], location="right",
                         shrink=0.78, pad=0.015, aspect=22)
    cb12.ax.tick_params(labelsize=9)
    cb12.set_label("error", fontsize=10)
    # Separate colorbar for panel 2
    cb3 = fig.colorbar(im2, ax=axes[2], location="right",
                        shrink=0.78, pad=0.015, aspect=22)
    cb3.ax.tick_params(labelsize=9)
    cb3.set_label("reduction", fontsize=10)

    # Title strip: title (16pt) + annotation (13pt) on phantom axes
    title_ax.text(
        0.5, 0.66,
        f"CARE-KV vs base_quant — {kind} error decomposition, layer {layer_id}",
        ha="center", va="center", fontsize=16,
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.5, 0.20,
        f"mean error: {stats['base_mean']:.4f} → {stats['care_mean']:.4f},   "
        f"reduction {stats['relative_error_redux']*100:.1f}%,   "
        f"recovered {stats['recovered_sparsity']*100:.1f}%",
        ha="center", va="center", fontsize=13, color="#444444",
        transform=title_ax.transAxes,
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_2d_paper_combined_KV(
    K_data, V_data, layer_id, out_path,
    error_cmap, positive_reduction_cmap,
    error_percentile, reduction_percentile, reduction_gamma,
    mask_contour_top_percent,
):
    """2-row × 3-col combined paper figure for one representative layer.

    Row 1: K (base / after-CARE / reduction) — shared K colorbars
    Row 2: V (base / after-CARE / reduction) — shared V colorbars
    K and V are NOT scale-matched (their magnitudes differ ~50x).
    """
    fig = plt.figure(figsize=(16, 9.2), layout="constrained")
    gs = fig.add_gridspec(3, 3,
                          height_ratios=[0.10, 1.0, 1.0],
                          hspace=0.02)
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    row_axes = [
        [fig.add_subplot(gs[r, c]) for c in range(3)]
        for r in (1, 2)
    ]

    for row, (kind, data) in enumerate([("K", K_data), ("V", V_data)]):
        axes = row_axes[row]
        err_base, err_care, err_reduction, recovered = (
            data["err_base"], data["err_care"],
            data["err_reduction"], data["recovered"]
        )
        err_vmax = _percentile(err_base, error_percentile)
        pos_red = np.clip(err_reduction, 0, None)
        pos_nonzero = pos_red[pos_red > 0]
        pos_vmax = _percentile(pos_nonzero, reduction_percentile, default=1.0) \
                   if pos_nonzero.size > 0 else 1.0
        if reduction_gamma and reduction_gamma > 0 and reduction_gamma != 1.0:
            pos_norm = PowerNorm(gamma=reduction_gamma, vmin=0.0, vmax=pos_vmax)
        else:
            pos_norm = None

        im0 = axes[0].imshow(err_base, cmap=error_cmap,
                              vmin=0.0, vmax=err_vmax,
                              aspect="auto", interpolation="nearest")
        axes[0].set_title(f"{kind}: Base INT3 error", fontsize=13)
        im1 = axes[1].imshow(err_care, cmap=error_cmap,
                              vmin=0.0, vmax=err_vmax,
                              aspect="auto", interpolation="nearest")
        axes[1].set_title(f"{kind}: After CARE-KV", fontsize=13)
        if pos_norm is not None:
            im2 = axes[2].imshow(pos_red, cmap=positive_reduction_cmap,
                                  norm=pos_norm,
                                  aspect="auto", interpolation="nearest")
        else:
            im2 = axes[2].imshow(pos_red, cmap=positive_reduction_cmap,
                                  vmin=0.0, vmax=pos_vmax,
                                  aspect="auto", interpolation="nearest")
        axes[2].set_title(f"{kind}: Error reduced", fontsize=13)

        if mask_contour_top_percent > 0:
            hi_mask = _paper_high_recovery_mask(recovered, mask_contour_top_percent)
            if hi_mask.any():
                axes[2].contour(hi_mask.astype(float), levels=[0.5],
                                colors="#8B0000", linewidths=0.35, alpha=0.4)
        axes[2].text(
            0.02, 0.98, "positive = improved",
            transform=axes[2].transAxes,
            fontsize=8, color="#333", alpha=0.8,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.6),
        )

        # Axis label discipline (per row): Token only on col 0; Channel on all
        for ax in axes:
            ax.set_xlabel("Channel", fontsize=11)
            ax.tick_params(labelsize=9)
        axes[0].set_ylabel("Token", fontsize=11)
        for ax in axes[1:]:
            ax.set_ylabel("")
            ax.tick_params(labelleft=True, labelsize=9)

        cb12 = fig.colorbar(im1, ax=axes[:2], location="right",
                             shrink=0.85, pad=0.015, aspect=22)
        cb12.ax.tick_params(labelsize=9)
        cb12.set_label(f"{kind} error", fontsize=10)
        cb3 = fig.colorbar(im2, ax=axes[2], location="right",
                            shrink=0.85, pad=0.015, aspect=22)
        cb3.ax.tick_params(labelsize=9)
        cb3.set_label(f"{kind} reduction", fontsize=10)

    title_ax.text(
        0.5, 0.50,
        f"CARE-KV vs base_quant — K (top) / V (bottom), layer {layer_id}",
        ha="center", va="center", fontsize=16,
        transform=title_ax.transAxes,
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11, 21])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-channels", type=int, default=128)
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--k-channel-group", type=int, default=32)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--store-abs-k", type=int, default=2)
    ap.add_argument("--store-abs-v", type=int, default=4)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--stats-json", default=None)

    ap.add_argument("--plot-mode",
                    choices=["error-decomposition", "visible-error",
                             "clean-error", "paper-clean", "both", "all"],
                    default="error-decomposition")
    ap.add_argument("--paper-combined-layer", type=int, default=11,
                    help="paper-clean: which layer goes into the combined K/V figure")
    ap.add_argument("--reduction-percentile", type=float, default=97.0,
                    help="paper-clean: vmax = percentile(positive_reduction>0, --reduction-percentile)")
    ap.add_argument("--reduction-gamma", type=float, default=0.5,
                    help="paper-clean: PowerNorm gamma for panel 3 (0.5 = boost small values)")
    ap.add_argument("--mask-contour-top-percent", type=float, default=2.0,
                    help="paper-clean: contour only the top X%% of recovered cells on panel 3")
    ap.add_argument("--error-cmap",     default="inferno")
    ap.add_argument("--reduction-cmap", default="RdBu_r")
    ap.add_argument("--residual-cmap",  default="hot")
    ap.add_argument("--error-percentile", type=float, default=99.0)
    ap.add_argument("--error-gain",       type=float, default=1.0)
    ap.add_argument("--log-error", action="store_true",
                    help="off by default; pass to plot log1p(error*gain)")
    ap.add_argument("--overlay-recovered-mask", action="store_true",
                    help="visible-error: scatter top recovered cells over |X_fp|")
    ap.add_argument("--overlay-top-percent", type=float, default=1.0,
                    help="overlay/scatter top X%% of recovered cells (default 1.0)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[care-err] model={args.model_id} device={device} bits={args.base_bits} "
          f"mode={args.plot_mode}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    m = LlamaForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16,
        device_map=device if device == "cuda" else None,
    )
    m.eval()
    cfg = m.config
    Hkv = cfg.num_key_value_heads
    Hd  = cfg.hidden_size // cfg.num_attention_heads
    assert Hd % args.k_channel_group == 0

    text = args.prompt or ("The quick brown fox jumps over the lazy dog. " * 200)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    input_ids = enc["input_ids"].to(device)
    T = int(input_ids.shape[1])
    print(f"[care-err] tokens={T} Hkv={Hkv} Hd={Hd}", flush=True)

    with torch.no_grad():
        out = m.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states

    rotary_emb = (m.model.rotary_emb if hasattr(m.model, "rotary_emb")
                  else m.model.layers[0].self_attn.rotary_emb)
    position_ids = torch.arange(T, device=device).unsqueeze(0)

    os.makedirs(args.out_dir, exist_ok=True)
    qcfg = QuantConfig(bits=args.base_bits, group_size=args.k_channel_group)

    layer_stats = {}
    # For paper-clean combined figure: cache (layer, kind) arrays we need later
    paper_cache = {}

    for layer_id in args.layers:
        if layer_id >= cfg.num_hidden_layers:
            print(f"[care-err] skip layer {layer_id} (only {cfg.num_hidden_layers} layers)",
                  flush=True)
            continue
        with torch.no_grad():
            h = hidden_states[layer_id]
            layer = m.model.layers[layer_id]
            h_n = layer.input_layernorm(h)
            K_pre = layer.self_attn.k_proj(h_n)
            V_lin = layer.self_attn.v_proj(h_n)
            B = K_pre.shape[0]
            K_pre_h = K_pre.reshape(B, T, Hkv, Hd).transpose(1, 2)
            V_h     = V_lin.reshape(B, T, Hkv, Hd).transpose(1, 2)
            cos, sin = rotary_emb(V_h, position_ids)
            _, K_post_h = apply_rotary_pos_emb(V_h, K_pre_h, cos, sin)

            K_fp = K_post_h[0].float()
            V_fp = V_h[0].float()

            _, _, R_K, _ = quantize_and_residual(K_fp, qcfg)
            _, _, R_V, _ = quantize_and_residual(V_fp, qcfg)
            K_hat = K_fp - R_K
            V_hat = V_fp - R_V

            K_mask = _select_k_stored_mask(R_K, args.page_size,
                                           args.k_channel_group, args.store_abs_k)
            V_mask = _select_v_stored_mask(R_V, args.page_size, args.store_abs_v)
            K_care = K_hat + R_K * K_mask.float()
            V_care = V_hat + R_V * V_mask.float()

            def _flat(t):
                return (t.transpose(0, 1).reshape(T, Hkv * Hd)
                        .float().detach().cpu().numpy())
            K_fp_n, K_hat_n, K_care_n = _flat(K_fp), _flat(K_hat), _flat(K_care)
            V_fp_n, V_hat_n, V_care_n = _flat(V_fp), _flat(V_hat), _flat(V_care)
            K_mask_n = _flat(K_mask.float()) > 0.5
            V_mask_n = _flat(V_mask.float()) > 0.5

        for kind, X_fp, X_hat, X_care, mask_n in [
            ("K", K_fp_n, K_hat_n, K_care_n, K_mask_n),
            ("V", V_fp_n, V_hat_n, V_care_n, V_mask_n),
        ]:
            X_fp_abs      = np.abs(X_fp)
            err_base      = np.abs(X_fp - X_hat)
            err_care      = np.abs(X_fp - X_care)
            err_reduction = err_base - err_care
            recovered     = np.abs(X_care - X_hat)

            stats = {
                "base_mean":              float(err_base.mean()),
                "base_max":               float(err_base.max()),
                "care_mean":              float(err_care.mean()),
                "care_max":               float(err_care.max()),
                "rmse_base":              float(np.sqrt((err_base ** 2).mean())),
                "rmse_care":              float(np.sqrt((err_care ** 2).mean())),
                "redux_mean":             float(err_reduction.mean()),
                "redux_max":              float(err_reduction.max()),
                "relative_error_redux":   float(err_reduction.mean()
                                                / max(err_base.mean(), 1e-12)),
                "recovered_sparsity":     float(mask_n.mean()),
                "recovered_count":        int(mask_n.sum()),
                "positive_reduction_ratio": float((err_base > err_care).mean()),
            }
            layer_stats[f"layer{layer_id:02d}_{kind}"] = stats
            print(f"[care-err] layer {layer_id:02d} {kind}: "
                  f"base mean={stats['base_mean']:.4f} care mean={stats['care_mean']:.4f} "
                  f"rel_redux={stats['relative_error_redux']*100:.1f}% "
                  f"count={stats['recovered_count']} "
                  f"pos_redux={stats['positive_reduction_ratio']:.3f}",
                  flush=True)

            do_decomp = args.plot_mode in ("error-decomposition", "both", "all", "clean-error")
            do_visible = args.plot_mode in ("visible-error", "both", "all")
            do_clean3d = args.plot_mode in ("clean-error", "all")
            do_decomp_3d = args.plot_mode in ("error-decomposition", "both", "all")
            do_paper_clean = args.plot_mode in ("paper-clean", "all")

            if do_decomp_3d:
                p = os.path.join(args.out_dir,
                                 f"3d_carekv_{kind}_error_layer{layer_id:02d}.png")
                _plot_3d_error_4panel(
                    err_base, err_care, err_reduction, recovered,
                    layer_id, kind, p,
                    args.max_tokens, args.max_channels,
                    args.error_cmap, args.reduction_cmap, args.residual_cmap,
                    args.error_percentile, args.error_gain, args.log_error,
                )
                print(f"[care-err] saved {p}", flush=True)

            if do_decomp:
                # 5-panel 2D heatmap is the PRIMARY paper figure; always
                # produced for error-decomposition and clean-error modes.
                p = os.path.join(args.out_dir,
                                 f"heatmap_carekv_{kind}_error_layer{layer_id:02d}.png")
                _plot_2d_heatmap_primary(
                    err_base, err_care, err_reduction, recovered, mask_n,
                    layer_id, kind, p, stats,
                    args.error_cmap, args.reduction_cmap, args.residual_cmap,
                    args.error_percentile, args.error_gain, args.log_error,
                )
                print(f"[care-err] saved {p}", flush=True)

            if do_visible:
                p = os.path.join(args.out_dir,
                                 f"3d_carekv_{kind}_visible_layer{layer_id:02d}.png")
                _plot_3d_visible_5panel(
                    X_fp_abs, err_base, err_care, err_reduction, recovered,
                    layer_id, kind, p,
                    args.max_tokens, args.max_channels,
                    args.error_cmap, args.reduction_cmap, args.residual_cmap,
                    args.error_percentile, args.error_gain, args.log_error,
                    args.overlay_recovered_mask, args.overlay_top_percent,
                )
                print(f"[care-err] saved {p}", flush=True)
                p = os.path.join(args.out_dir,
                                 f"heatmap_carekv_{kind}_visible_layer{layer_id:02d}.png")
                _plot_2d_heatmap_5panel_visible(
                    X_fp_abs, err_base, err_care, err_reduction, recovered, mask_n,
                    layer_id, kind, p,
                    args.error_cmap, args.reduction_cmap, args.residual_cmap,
                    args.error_percentile, args.error_gain, args.log_error,
                    args.overlay_recovered_mask, args.overlay_top_percent,
                )
                print(f"[care-err] saved {p}", flush=True)

            if do_clean3d:
                p = os.path.join(args.out_dir,
                                 f"3d_carekv_{kind}_error_clean_layer{layer_id:02d}.png")
                _plot_3d_clean_3panel(
                    err_base, err_care, recovered,
                    layer_id, kind, p,
                    args.max_tokens, args.max_channels,
                    args.error_cmap, args.residual_cmap,
                    args.error_percentile, args.error_gain, args.log_error,
                    args.overlay_top_percent,
                )
                print(f"[care-err] saved {p}", flush=True)

            if do_paper_clean:
                POS_CHOICES = {"Reds", "YlOrRd", "OrRd", "Oranges", "YlGnBu"}
                pos_cmap = args.reduction_cmap if args.reduction_cmap in POS_CHOICES else "Reds"
                p = os.path.join(args.out_dir,
                                 f"paper_carekv_{kind}_error_layer{layer_id:02d}.png")
                _plot_2d_paper_clean_3panel(
                    err_base, err_care, err_reduction, recovered, mask_n,
                    layer_id, kind, p, stats,
                    args.error_cmap, pos_cmap,
                    args.error_percentile, args.reduction_percentile,
                    args.reduction_gamma, args.mask_contour_top_percent,
                )
                print(f"[care-err] saved {p}", flush=True)
                if layer_id == args.paper_combined_layer:
                    paper_cache[kind] = dict(
                        err_base=err_base, err_care=err_care,
                        err_reduction=err_reduction,
                        recovered=recovered, mask=mask_n,
                    )

    if args.plot_mode in ("paper-clean", "all") and \
       "K" in paper_cache and "V" in paper_cache:
        POS_CHOICES = {"Reds", "YlOrRd", "OrRd", "Oranges", "YlGnBu"}
        pos_cmap = args.reduction_cmap if args.reduction_cmap in POS_CHOICES else "Reds"
        p = os.path.join(args.out_dir,
                         f"paper_carekv_KV_error_layer{args.paper_combined_layer:02d}.png")
        _plot_2d_paper_combined_KV(
            paper_cache["K"], paper_cache["V"],
            args.paper_combined_layer, p,
            args.error_cmap, pos_cmap,
            args.error_percentile, args.reduction_percentile,
            args.reduction_gamma, args.mask_contour_top_percent,
        )
        print(f"[care-err] saved {p}", flush=True)

    if args.stats_json:
        with open(args.stats_json, "w") as f:
            json.dump(layer_stats, f, indent=2)
        print(f"[care-err] wrote stats → {args.stats_json}", flush=True)


if __name__ == "__main__":
    main()
