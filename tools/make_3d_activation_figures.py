"""tools/make_3d_activation_figures.py

Phase K-c — 3D activation distribution figures (Channel × Token × |value|)
for K pre-RoPE, K post-RoPE, and V at selected layers.

Mirrors the classic SmoothQuant / LLM.int8()-style outlier visualization
but rendered with matplotlib's plot_surface for smoothness on the
TinyLlama scale (T up to ~512, channels = num_kv_heads × head_dim).

Outputs:
  <out-dir>/3d_activation_layer{NN}.png    — one combined figure per layer
                                              (3 subplots: K_pre / K_post / V)
  <out-dir>/3d_data/layer{NN}_{k_pre,k_post,v}.npy
                                            — full-resolution |value| arrays
                                              (T, num_kv_heads × head_dim)

Usage:
  PYTHONPATH=/home/soeun python tools/make_3d_activation_figures.py \
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


def _stride_downsample(arr: np.ndarray, max_dim: int, axis: int) -> np.ndarray:
    n = arr.shape[axis]
    if n <= max_dim:
        return arr
    step = max(1, n // max_dim)
    idx = np.arange(0, n, step)[:max_dim]
    return np.take(arr, idx, axis=axis)


def _plot_layer(k_pre: np.ndarray, k_post: np.ndarray, v: np.ndarray,
                layer_id: int, out_path: str,
                max_tokens: int, max_channels: int) -> None:
    panels = [
        (f"Layer {layer_id} Keys (pre-RoPE)",  k_pre),
        (f"Layer {layer_id} Keys (post-RoPE)", k_post),
        (f"Layer {layer_id} Values",           v),
    ]
    fig = plt.figure(figsize=(18, 6))
    for i, (title, arr) in enumerate(panels):
        ds = _stride_downsample(arr, max_tokens, axis=0)
        ds = _stride_downsample(ds, max_channels, axis=1)
        T, C = ds.shape
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        X, Y = np.meshgrid(np.arange(C), np.arange(T))
        rstride = max(1, T // 64)
        cstride = max(1, C // 64)
        surf = ax.plot_surface(
            X, Y, ds, cmap="viridis", linewidth=0, antialiased=False,
            rstride=rstride, cstride=cstride,
        )
        ax.set_xlabel("Channel")
        ax.set_ylabel("Token")
        ax.set_zlabel("|value|")
        ax.set_title(title, fontsize=11)
        ax.view_init(elev=28, azim=-58)
        fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 11, 21])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-channels", type=int, default=512)
    ap.add_argument("--prompt", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3d-fig] model={args.model_id} device={device}", flush=True)

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

    # Build a seq_len-token prompt — repeat a pangram to fill the window
    text = args.prompt or ("The quick brown fox jumps over the lazy dog. " * 200)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len)
    input_ids = enc["input_ids"].to(device)
    T_actual = int(input_ids.shape[1])
    print(f"[3d-fig] tokens={T_actual} Hkv={Hkv} Hd={Hd}", flush=True)

    with torch.no_grad():
        out = m.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = out.hidden_states  # tuple len=num_hidden_layers+1, each (B,T,D)

    os.makedirs(args.out_dir, exist_ok=True)
    data_dir = os.path.join(args.out_dir, "3d_data")
    os.makedirs(data_dir, exist_ok=True)

    # Rotary embedding — try multiple locations across transformers versions
    rotary_emb = None
    if hasattr(m.model, "rotary_emb"):
        rotary_emb = m.model.rotary_emb
    elif hasattr(m.model.layers[0].self_attn, "rotary_emb"):
        rotary_emb = m.model.layers[0].self_attn.rotary_emb
    else:
        raise RuntimeError("could not locate LlamaRotaryEmbedding on model or first layer")

    position_ids = torch.arange(T_actual, device=device).unsqueeze(0)

    for layer_id in args.layers:
        if layer_id >= cfg.num_hidden_layers:
            print(f"[3d-fig] skip layer {layer_id} (only {cfg.num_hidden_layers} layers)",
                  flush=True)
            continue
        with torch.no_grad():
            h = hidden_states[layer_id]  # input to layer `layer_id` (residual stream)
            layer = m.model.layers[layer_id]
            h_n = layer.input_layernorm(h)
            K_pre = layer.self_attn.k_proj(h_n)  # (B, T, Hkv*Hd)
            V_lin = layer.self_attn.v_proj(h_n)
            B, T, _ = K_pre.shape
            K_pre_h  = K_pre.reshape(B, T, Hkv, Hd).transpose(1, 2)   # (B, Hkv, T, Hd)
            V_h      = V_lin.reshape(B, T, Hkv, Hd).transpose(1, 2)
            cos, sin = rotary_emb(V_h, position_ids)
            # apply_rotary_pos_emb wants (q, k, cos, sin) — pass V as a throwaway q
            _, K_post_h = apply_rotary_pos_emb(V_h, K_pre_h, cos, sin)

            def _flat(t):
                return (t.transpose(1, 2).reshape(B, T, Hkv * Hd)[0]
                        .abs().float().detach().cpu().numpy())
            K_pre_flat  = _flat(K_pre_h)
            K_post_flat = _flat(K_post_h)
            V_flat      = _flat(V_h)

        np.save(os.path.join(data_dir, f"layer{layer_id:02d}_k_pre.npy"),  K_pre_flat)
        np.save(os.path.join(data_dir, f"layer{layer_id:02d}_k_post.npy"), K_post_flat)
        np.save(os.path.join(data_dir, f"layer{layer_id:02d}_v.npy"),      V_flat)

        print(f"[3d-fig] layer {layer_id:02d}: shape={K_pre_flat.shape}  "
              f"|K_pre|max={K_pre_flat.max():.3f}  "
              f"|K_post|max={K_post_flat.max():.3f}  "
              f"|V|max={V_flat.max():.3f}",
              flush=True)

        out_png = os.path.join(args.out_dir, f"3d_activation_layer{layer_id:02d}.png")
        _plot_layer(K_pre_flat, K_post_flat, V_flat, layer_id, out_png,
                    args.max_tokens, args.max_channels)
        print(f"[3d-fig] saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
