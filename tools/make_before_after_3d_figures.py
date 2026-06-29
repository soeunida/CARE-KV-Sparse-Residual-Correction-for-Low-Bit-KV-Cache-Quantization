"""tools/make_before_after_3d_figures.py

Phase K-d — CARE-KV before/after 3D distribution diagnostics.

For each selected layer, generates a single figure with **6 subplots**
arranged in two rows showing what the K and V activations look like in
three regimes:

  Row 1 (Keys, post-RoPE):
    K_fp           — original fp16 post-RoPE K (what fp16 attention sees)
    K_hat          — dequantize(quantize(K_fp))            (base_quant INT3)
    K_care_store   — K_hat + selected/stored K residuals   (CARE-KV INT3)

  Row 2 (Values):
    V_fp           — original fp16 V
    V_hat          — dequantize(quantize(V_fp))            (base_quant INT3)
    V_care_store   — V_hat + selected/stored V residuals   (CARE-KV INT3)

Each subplot is a `plot_surface` over (Channel x Token x |value|).

Store-time selection mirrors CARE-KV's paper-best config:
  page_size=16, k_channel_group=32, v_token_block=4,
  STORE_ABS_K=2 (per page, per kv_head, per channel_group),
  STORE_ABS_V=4 (per page, per kv_head),
  scored by L2 norm of the residual segment.

Outputs:
  <out-dir>/3d_before_after_layer{NN}.png
  <out-dir>/3d_data/layer{NN}_{k_fp,k_hat,k_care,v_fp,v_hat,v_care}.npy

Usage:
  PYTHONPATH=/home/soeun python tools/make_before_after_3d_figures.py \
      --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --out-dir results/paper_eval_20260529_015053/figures \
      --layers 0 11 21 --seq-len 512 \
      --max-tokens 256 --max-channels 512
"""
from __future__ import annotations
import argparse, os, sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from CARE_KV.care_kv.quantizer import QuantConfig, quantize_and_residual


# ─────────────────────────────────────────────
# Store-time residual selection (paper-best knobs)
# ─────────────────────────────────────────────

def _select_k_stored_mask(R_K: torch.Tensor, page_size: int,
                          k_channel_group: int, store_abs_k: int) -> torch.Tensor:
    """
    R_K: (Hkv, T, D)   K residual per kv-head, per token, per channel
    Returns boolean mask same shape; True where the residual is stored.

    Selection: per (page, kv_head, channel_group), pick the top `store_abs_k`
    tokens by L2 norm of the (group,)-vector residual.
    """
    Hkv, T, D = R_K.shape
    assert D % k_channel_group == 0
    G = D // k_channel_group
    mask = torch.zeros_like(R_K, dtype=torch.bool)
    for p_start in range(0, T, page_size):
        p_end = min(p_start + page_size, T)
        for h in range(Hkv):
            for g in range(G):
                c0, c1 = g * k_channel_group, (g + 1) * k_channel_group
                segs = R_K[h, p_start:p_end, c0:c1]               # (page_len, k_channel_group)
                scores = segs.float().norm(dim=-1)                # (page_len,)
                k = min(store_abs_k, segs.shape[0])
                top_idx = torch.topk(scores, k).indices            # (k,)
                mask[h, p_start + top_idx, c0:c1] = True
    return mask


def _select_v_stored_mask(R_V: torch.Tensor, page_size: int,
                          store_abs_v: int) -> torch.Tensor:
    """
    R_V: (Hkv, T, D)   V residual per kv-head, per token, per channel
    Returns boolean mask same shape; True where the residual is stored.

    Selection: per (page, kv_head), pick top `store_abs_v` tokens
    by L2 norm of the full residual vector (all D channels).
    A stored token has its full residual restored across all channels.
    """
    Hkv, T, D = R_V.shape
    mask = torch.zeros_like(R_V, dtype=torch.bool)
    for p_start in range(0, T, page_size):
        p_end = min(p_start + page_size, T)
        page = R_V[:, p_start:p_end, :]                            # (Hkv, page_len, D)
        scores = page.float().norm(dim=-1)                         # (Hkv, page_len)
        for h in range(Hkv):
            k = min(store_abs_v, page.shape[1])
            top_idx = torch.topk(scores[h], k).indices              # (k,)
            mask[h, p_start + top_idx, :] = True
    return mask


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def _stride_downsample(arr: np.ndarray, max_dim: int, axis: int) -> np.ndarray:
    n = arr.shape[axis]
    if n <= max_dim:
        return arr
    step = max(1, n // max_dim)
    idx = np.arange(0, n, step)[:max_dim]
    return np.take(arr, idx, axis=axis)


def _plot_one_panel(ax, arr: np.ndarray, title: str, vmax: float,
                    max_tokens: int, max_channels: int) -> None:
    ds = _stride_downsample(arr, max_tokens, axis=0)
    ds = _stride_downsample(ds, max_channels, axis=1)
    T, C = ds.shape
    X, Y = np.meshgrid(np.arange(C), np.arange(T))
    rstride = max(1, T // 64)
    cstride = max(1, C // 64)
    surf = ax.plot_surface(
        X, Y, ds, cmap="viridis", linewidth=0, antialiased=False,
        rstride=rstride, cstride=cstride, vmin=0, vmax=vmax,
    )
    ax.set_xlabel("Channel"); ax.set_ylabel("Token"); ax.set_zlabel("|value|")
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=28, azim=-58)
    ax.set_zlim(0, vmax)
    return surf


def _plot_layer(K_fp, K_hat, K_care, V_fp, V_hat, V_care,
                layer_id: int, out_path: str,
                max_tokens: int, max_channels: int) -> None:
    # Shared z-axis scale per row so visual comparison is honest
    k_vmax = max(K_fp.max(), K_hat.max(), K_care.max())
    v_vmax = max(V_fp.max(), V_hat.max(), V_care.max())

    fig = plt.figure(figsize=(18, 12))
    panels = [
        (1, f"Layer {layer_id}  K_fp           (post-RoPE, fp16)",       K_fp,    k_vmax),
        (2, f"Layer {layer_id}  K_hat          (base_quant INT3)",        K_hat,   k_vmax),
        (3, f"Layer {layer_id}  K_care_store   (CARE-KV INT3 + R_K)",     K_care,  k_vmax),
        (4, f"Layer {layer_id}  V_fp           (fp16)",                   V_fp,    v_vmax),
        (5, f"Layer {layer_id}  V_hat          (base_quant INT3)",        V_hat,   v_vmax),
        (6, f"Layer {layer_id}  V_care_store   (CARE-KV INT3 + R_V)",     V_care,  v_vmax),
    ]
    for idx, title, arr, vmax in panels:
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        surf = _plot_one_panel(ax, arr, title, vmax, max_tokens, max_channels)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    fig.suptitle(
        f"CARE-KV before / after — Layer {layer_id}   "
        f"(K shared z-axis up to {k_vmax:.2f};  V shared z-axis up to {v_vmax:.2f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=110)
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
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-channels", type=int, default=512)
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--k-channel-group", type=int, default=32)
    ap.add_argument("--page-size", type=int, default=16)
    ap.add_argument("--store-abs-k", type=int, default=2)
    ap.add_argument("--store-abs-v", type=int, default=4)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bef-aft] model={args.model_id} device={device} bits={args.base_bits}",
          flush=True)

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
    assert Hd % args.k_channel_group == 0, \
        f"head_dim {Hd} must be divisible by k_channel_group {args.k_channel_group}"

    text = args.prompt or ("The quick brown fox jumps over the lazy dog. " * 200)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    input_ids = enc["input_ids"].to(device)
    T = int(input_ids.shape[1])
    print(f"[bef-aft] tokens={T} Hkv={Hkv} Hd={Hd}", flush=True)

    with torch.no_grad():
        out = m.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states

    rotary_emb = (m.model.rotary_emb if hasattr(m.model, "rotary_emb")
                  else m.model.layers[0].self_attn.rotary_emb)
    position_ids = torch.arange(T, device=device).unsqueeze(0)

    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = os.path.join(args.out_dir, "3d_data")
    os.makedirs(data_dir, exist_ok=True)

    qcfg_k = QuantConfig(bits=args.base_bits, group_size=args.k_channel_group)
    qcfg_v = QuantConfig(bits=args.base_bits, group_size=args.k_channel_group)

    for layer_id in args.layers:
        if layer_id >= cfg.num_hidden_layers:
            print(f"[bef-aft] skip layer {layer_id} (only {cfg.num_hidden_layers} layers)",
                  flush=True)
            continue
        with torch.no_grad():
            h = hidden_states[layer_id]
            layer = m.model.layers[layer_id]
            h_n = layer.input_layernorm(h)
            K_pre = layer.self_attn.k_proj(h_n)
            V_lin = layer.self_attn.v_proj(h_n)
            B = K_pre.shape[0]
            K_pre_h = K_pre.reshape(B, T, Hkv, Hd).transpose(1, 2)   # (B, Hkv, T, Hd)
            V_h     = V_lin.reshape(B, T, Hkv, Hd).transpose(1, 2)
            cos, sin = rotary_emb(V_h, position_ids)
            _, K_post_h = apply_rotary_pos_emb(V_h, K_pre_h, cos, sin)  # (B, Hkv, T, Hd)

            # Strip batch dim → (Hkv, T, Hd) in fp32 for quantization
            K_fp = K_post_h[0].float()
            V_fp = V_h[0].float()

            # Base quantize (round-trip) → K_hat / V_hat and residuals
            _, _, R_K, _ = quantize_and_residual(K_fp, qcfg_k)
            _, _, R_V, _ = quantize_and_residual(V_fp, qcfg_v)
            K_hat = K_fp - R_K
            V_hat = V_fp - R_V

            # Store-time selection mask (paper-best knobs)
            K_mask = _select_k_stored_mask(R_K, args.page_size,
                                           args.k_channel_group, args.store_abs_k)
            V_mask = _select_v_stored_mask(R_V, args.page_size, args.store_abs_v)

            K_care = K_hat + R_K * K_mask.float()
            V_care = V_hat + R_V * V_mask.float()

            def _flat(t):
                # (Hkv, T, Hd) → (T, Hkv*Hd)
                return (t.transpose(0, 1).reshape(T, Hkv * Hd)
                        .abs().float().detach().cpu().numpy())
            K_fp_a, K_hat_a, K_care_a = _flat(K_fp), _flat(K_hat), _flat(K_care)
            V_fp_a, V_hat_a, V_care_a = _flat(V_fp), _flat(V_hat), _flat(V_care)

            # Selected-slot coverage (sanity)
            k_kept = float(K_mask.float().mean())
            v_kept = float(V_mask.float().mean())

        # Save raw arrays
        for name, arr in [("k_fp", K_fp_a), ("k_hat", K_hat_a), ("k_care", K_care_a),
                          ("v_fp", V_fp_a), ("v_hat", V_hat_a), ("v_care", V_care_a)]:
            np.save(os.path.join(data_dir, f"layer{layer_id:02d}_{name}.npy"), arr)

        # Per-cell magnitudes for the summary
        print(f"[bef-aft] layer {layer_id:02d}: "
              f"|K_fp|max={K_fp_a.max():.3f}  "
              f"|K_hat|max={K_hat_a.max():.3f}  "
              f"|K_care|max={K_care_a.max():.3f}    "
              f"|V_fp|max={V_fp_a.max():.3f}  "
              f"|V_hat|max={V_hat_a.max():.3f}  "
              f"|V_care|max={V_care_a.max():.3f}    "
              f"K_kept={k_kept:.3f}  V_kept={v_kept:.3f}",
              flush=True)
        # Reconstruction error
        k_err_base = float(((K_fp_a - K_hat_a) ** 2).mean() ** 0.5)
        k_err_care = float(((K_fp_a - K_care_a) ** 2).mean() ** 0.5)
        v_err_base = float(((V_fp_a - V_hat_a) ** 2).mean() ** 0.5)
        v_err_care = float(((V_fp_a - V_care_a) ** 2).mean() ** 0.5)
        print(f"[bef-aft] layer {layer_id:02d}: "
              f"K RMSE base={k_err_base:.5f} → care={k_err_care:.5f} "
              f"(-{(1 - k_err_care/max(k_err_base,1e-12))*100:.1f}%)    "
              f"V RMSE base={v_err_base:.5f} → care={v_err_care:.5f} "
              f"(-{(1 - v_err_care/max(v_err_base,1e-12))*100:.1f}%)",
              flush=True)

        out_png = os.path.join(args.out_dir, f"3d_before_after_layer{layer_id:02d}.png")
        _plot_layer(K_fp_a, K_hat_a, K_care_a, V_fp_a, V_hat_a, V_care_a,
                    layer_id, out_png, args.max_tokens, args.max_channels)
        print(f"[bef-aft] saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
