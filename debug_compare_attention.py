import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM

from CARE_KV.care_kv import CacheConfig, CAREKVCache, CAREKVLayer


MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LAYER_ID = int(os.environ.get("LAYER_ID", "0"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "64"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)

print(f"MODEL_ID={MODEL_ID}")
print(f"LAYER_ID={LAYER_ID}")
print(f"SEQ_LEN={SEQ_LEN}")
print(f"DEVICE={DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

model.eval()

cfg = model.config
layer = model.model.layers[LAYER_ID]
attn = layer.self_attn

prompt = "Explain KV cache quantization briefly. " * 20
enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
input_ids = enc["input_ids"].to(next(model.parameters()).device)
T = input_ids.shape[1]

print(f"Actual T={T}")

# Build layer input exactly like LLaMA layer input: embedding -> first layer input norm.
hidden = model.model.embed_tokens(input_ids)
hidden_norm = layer.input_layernorm(hidden)

position_ids = torch.arange(T, device=hidden.device).unsqueeze(0)
cache_position = torch.arange(T, device=hidden.device)

# Additive causal mask: shape (B, 1, T, T)
min_val = torch.finfo(hidden_norm.dtype).min
causal = torch.full((T, T), min_val, device=hidden.device, dtype=hidden_norm.dtype)
causal = torch.triu(causal, diagonal=1)
attention_mask = causal.unsqueeze(0).unsqueeze(0)

# Original HF attention output.
orig_out = None
errors = []

call_variants = []

# Newer transformers may use external RoPE position_embeddings.
try:
    pos_emb = model.model.rotary_emb(hidden_norm, position_ids)
    call_variants.append({
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_value": None,
        "output_attentions": False,
        "use_cache": False,
        "cache_position": cache_position,
        "position_embeddings": pos_emb,
    })
except Exception as e:
    errors.append(f"rotary_emb variant build failed: {repr(e)}")

call_variants.append({
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "past_key_value": None,
    "output_attentions": False,
    "use_cache": False,
    "cache_position": cache_position,
})

call_variants.append({
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "past_key_value": None,
    "output_attentions": False,
    "use_cache": False,
})

call_variants.append({
    "attention_mask": None,
    "position_ids": position_ids,
    "past_key_value": None,
    "output_attentions": False,
    "use_cache": False,
})

for i, kwargs in enumerate(call_variants):
    try:
        out = attn(hidden_norm, **kwargs)
        orig_out = out[0]
        print(f"Original HF attention call succeeded with variant {i}")
        break
    except Exception as e:
        errors.append(f"variant {i}: {repr(e)}")

if orig_out is None:
    print("Failed to call original HF attention.")
    for e in errors:
        print(e)
    raise SystemExit(1)

# CAREKVLayer FP attention path.
head_dim = cfg.hidden_size // cfg.num_attention_heads

def largest_divisor_leq(n, limit):
    limit = min(n, limit)
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1

care_cfg = CacheConfig(
    num_layers=cfg.num_hidden_layers,
    num_heads=cfg.num_attention_heads,
    head_dim=head_dim,
    base_bits=int(os.environ.get("BASE_BITS", "4")),
    group_size=largest_divisor_leq(head_dim, 64),
    k_channel_group=largest_divisor_leq(head_dim, 32),
    store_budget_ratio=0.10,
    read_budget_ratio=0.03,
)

care_cfg.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)

care_layer = CAREKVLayer(
    care_cfg,
    layer_id=LAYER_ID,
    W_Q=attn.q_proj.weight.detach().float(),
    W_K=attn.k_proj.weight.detach().float(),
    W_V=attn.v_proj.weight.detach().float(),
    W_O=attn.o_proj.weight.detach().float(),
    device=hidden.device,
).to(hidden.device)

cache = CAREKVCache(care_cfg, hidden.device)

care_out = care_layer.prefill(
    cache,
    hidden_norm[0].float(),
    attention_mask=attention_mask,
    position_embeddings=pos_emb,
).unsqueeze(0).to(orig_out.dtype)

# Metrics.
diff = care_out - orig_out
cos = F.cosine_similarity(
    care_out.reshape(1, -1).float(),
    orig_out.reshape(1, -1).float(),
).item()

rel_l2 = diff.float().norm().item() / (orig_out.float().norm().item() + 1e-8)
max_abs = diff.float().abs().max().item()
mean_abs = diff.float().abs().mean().item()

print("=" * 80)
print("HF original attention vs CAREKVLayer.prefill()")
print(f"shape_orig={tuple(orig_out.shape)}")
print(f"shape_care={tuple(care_out.shape)}")
print(f"cosine={cos:.8f}")
print(f"rel_l2={rel_l2:.8f}")
print(f"max_abs={max_abs:.8f}")
print(f"mean_abs={mean_abs:.8f}")
print("=" * 80)

# Save result summary.
os.makedirs("results", exist_ok=True)
with open(f"results/attention_compare_layer{LAYER_ID}.txt", "w") as f:
    f.write(f"MODEL_ID={MODEL_ID}\n")
    f.write(f"LAYER_ID={LAYER_ID}\n")
    f.write(f"SEQ_LEN={T}\n")
    f.write(f"cosine={cos:.8f}\n")
    f.write(f"rel_l2={rel_l2:.8f}\n")
    f.write(f"max_abs={max_abs:.8f}\n")
    f.write(f"mean_abs={mean_abs:.8f}\n")
