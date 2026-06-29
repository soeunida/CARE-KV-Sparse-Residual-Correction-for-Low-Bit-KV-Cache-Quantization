"""
care_kv/llama_patch.py
-----------------------
HuggingFace LLaMA monkey-patch for CARE-KV.

Usage
-----
    from transformers import LlamaForCausalLM, LlamaConfig
    from care_kv.llama_patch import patch_llama_model, CAREKVLlamaForwardHook

    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    care_model = patch_llama_model(model, care_cfg)

    # Prefill
    outputs = care_model.generate(input_ids, max_new_tokens=128)

Design
------
We replace each LlamaAttention layer with a CAREKVLlamaAttention wrapper.
The wrapper:
  1. Intercepts forward() calls
  2. On first call (prefill), runs prefill() and initializes per-sequence cache
  3. On subsequent calls (decode), runs decode_step()
  4. Manages per-sequence CAREKVCache objects

For multi-sequence batches, one CAREKVCache is created per sequence.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Dict, Any
import math

from .cache import CAREKVCache, CacheConfig
from .layer import CAREKVLayer


class CAREKVLlamaAttention(nn.Module):
    """
    Wraps a HuggingFace LlamaAttention module and replaces its forward
    with CARE-KV prefill + decode logic.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        cfg: CacheConfig,
        layer_id: int,
        device: torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.device = device
        self.original_attn = original_attn  # kept for weight access

        # Extract weights from original attention
        # LlamaAttention uses q_proj, k_proj, v_proj, o_proj
        W_Q = original_attn.q_proj.weight.data.clone()   # (H*D, model_dim)
        W_K = original_attn.k_proj.weight.data.clone()
        W_V = original_attn.v_proj.weight.data.clone()
        W_O = original_attn.o_proj.weight.data.clone()   # (model_dim, H*D)

        self.care_layer = CAREKVLayer(
            cfg=cfg,
            layer_id=layer_id,
            W_Q=W_Q,
            W_K=W_K,
            W_V=W_V,
            W_O=W_O,
            device=device,
        )

        # Per-sequence cache storage
        # key: sequence hash or batch position index
        self._caches: Dict[int, CAREKVCache] = {}
        self._prefilled: Dict[int, bool] = {}

    def _get_or_create_cache(self, seq_id, device=None):
        """
        Get or create CAREKVCache for a sequence.

        device is required when using device_map="auto"; each wrapped layer may
        receive hidden_states on a different CUDA device.
        """
        cache_device = device if device is not None else self.device

        cache = self._caches.get(seq_id, None)
        if cache is None or getattr(cache, "device", None) != cache_device:
            self._caches[seq_id] = CAREKVCache(self.cfg, cache_device)
            self._prefilled[seq_id] = False

        return self._caches[seq_id]

    def reset_cache(self, seq_id: Optional[int] = None):
        """Reset cache for a specific sequence or all sequences."""
        if seq_id is None:
            self._caches.clear()
            self._prefilled.clear()
        else:
            self._caches.pop(seq_id, None)
            self._prefilled.pop(seq_id, None)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        """
        CARE-KV wrapper forward.

        CAREKV_RETURN=original:
            return original HuggingFace attention output.

        CAREKV_RETURN=care:
            return CAREKVLayer output.
        """
        return_mode = os.environ.get("CAREKV_RETURN", "care").lower()

        call_kwargs = dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        if cache_position is not None:
            call_kwargs["cache_position"] = cache_position

        if return_mode in {"original", "passthrough", "hf"}:
            try:
                return self.original_attn(hidden_states, **call_kwargs)
            except TypeError:
                call_kwargs.pop("cache_position", None)
                return self.original_attn(hidden_states, **call_kwargs)

        B, T, _ = hidden_states.shape
        position_embeddings = kwargs.get("position_embeddings", None)

        # RoPE embeddings are required for CAREKVLayer.prefill() to match LLaMA.
        # If unavailable, fall back to original HF attention.
        if position_embeddings is None:
            try:
                return self.original_attn(hidden_states, **call_kwargs)
            except TypeError:
                call_kwargs.pop("cache_position", None)
                return self.original_attn(hidden_states, **call_kwargs)

        if B == 1:
            seq_id = 0

            # Determine whether this call carries the full sequence (prefill)
            # or a single new token (decode).  Two cases both feed the prefill
            # branch:
            #   use_cache=False:  HF re-sends the full prompt every step.
            #   use_cache=True with an empty/zero-length past_key_value: we do
            #     not (yet) update HF's DynamicCache, so its get_seq_length()
            #     stays 0 and HF re-passes the full sequence — same as the
            #     use_cache=False case.  We must reset our internal CAREKV
            #     cache before each prefill so pages don't accumulate.
            hf_cache_len = 0
            if use_cache and past_key_value is not None:
                try:
                    hf_cache_len = past_key_value.get_seq_length(self.layer_id)
                except Exception:
                    try:
                        hf_cache_len = past_key_value.get_seq_length()
                    except Exception:
                        hf_cache_len = 0
            full_prefill = (T > 1) or (use_cache and hf_cache_len == 0 and T >= 1
                                       and not self._prefilled.get(seq_id, False))
            if full_prefill:
                self.reset_cache(seq_id)

            cache = self._get_or_create_cache(seq_id, hidden_states.device)

            if T > 1:
                self._prefilled[seq_id] = True
                output = self.care_layer.prefill(
                    cache,
                    hidden_states[0],
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                )
                output = output.unsqueeze(0)
            else:
                if not self._prefilled.get(seq_id, False):
                    self._prefilled[seq_id] = True
                    output = self.care_layer.prefill(
                        cache,
                        hidden_states[0],
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                    )
                    output = output.unsqueeze(0)
                else:
                    output = self.care_layer.decode_step(
                        cache, hidden_states[0],
                        position_embeddings=position_embeddings,
                    )
                    output = output.unsqueeze(0)

        else:
            outputs = []
            for b in range(B):
                seq_id = b
                cache = self._get_or_create_cache(seq_id, hidden_states.device)
                hidden_b = hidden_states[b]

                if T > 1:
                    self._prefilled[seq_id] = True
                    out_b = self.care_layer.prefill(
                        cache,
                        hidden_b,
                        attention_mask=attention_mask,
                        position_embeddings=position_embeddings,
                    )
                else:
                    if not self._prefilled.get(seq_id, False):
                        self._prefilled[seq_id] = True
                        out_b = self.care_layer.prefill(
                            cache,
                            hidden_b,
                            attention_mask=attention_mask,
                            position_embeddings=position_embeddings,
                        )
                    else:
                        out_b = self.care_layer.decode_step(
                            cache, hidden_b,
                            position_embeddings=position_embeddings,
                        )

                outputs.append(out_b)

            output = torch.stack(outputs, dim=0)

        # ── HF DynamicCache synchronization ───────────────────────────
        # When use_cache=True, HF reads past_key_value.get_seq_length() to
        # decide whether to send the full sequence or just the new token on
        # the next forward call.  Its DynamicCache derives that length from
        # `key_cache[layer_idx].shape[-2]`.  CAREKV stores real K/V in its
        # own buffers, but we still need to feed shape-correct zeros into
        # HF's cache so its length advances and subsequent calls send only
        # the incremental token.  We don't use these values for attention.
        if (use_cache and past_key_value is not None
                and hasattr(past_key_value, "update")):
            try:
                bsz = hidden_states.shape[0]
                Hkv = self.cfg.num_kv_heads
                D = self.cfg.head_dim
                dummy_k = hidden_states.new_zeros((bsz, Hkv, T, D))
                dummy_v = hidden_states.new_zeros((bsz, Hkv, T, D))
                # DynamicCache.update returns (key_concat, value_concat); we
                # discard the return value.
                past_key_value.update(dummy_k, dummy_v, self.layer_id)
            except Exception:
                # If the cache API differs in this transformers version we
                # silently skip the sync — generation will still be correct
                # but slower (re-prefills each step).
                pass

        # Return signature: (attn_output, attn_weights, past_key_value).
        return output, None, past_key_value



def patch_llama_model(
    model: nn.Module,
    care_cfg: CacheConfig,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Replace all LlamaAttention layers in model with CAREKVLlamaAttention.

    Parameters
    ----------
    model    : HuggingFace LlamaForCausalLM (or LlamaModel)
    care_cfg : CacheConfig with matching num_layers, num_heads, head_dim
    device   : target device (defaults to model's current device)

    Returns
    -------
    model with attention layers replaced (in-place modification)
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Find all attention layers
    # LlamaForCausalLM: model.model.layers[i].self_attn
    replaced = 0
    for name, module in model.named_modules():
        # Handle both LlamaAttention class name variants
        cls_name = type(module).__name__
        if "Attention" in cls_name and hasattr(module, "q_proj"):
            # Get parent module and attribute name
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]

            # Infer layer_id from name (e.g., "model.layers.0.self_attn")
            layer_id = 0
            for part in parts:
                if part.isdigit():
                    layer_id = int(part)
                    break

            care_attn = CAREKVLlamaAttention(
                original_attn=module,
                cfg=care_cfg,
                layer_id=layer_id,
                device=device,
            )
            setattr(parent, attr, care_attn)
            replaced += 1

    print(f"[CARE-KV] Replaced {replaced} attention layers with CARE-KV.")
    return model


def reset_all_caches(model: nn.Module):
    """Reset all CARE-KV caches in a patched model (call between generations)."""
    for module in model.modules():
        if isinstance(module, CAREKVLlamaAttention):
            module.reset_cache()
