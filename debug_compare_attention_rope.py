import os
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM

try:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
except Exception as e:
    raise RuntimeError(f"Failed to import LLaMA helpers: {e}")

MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
LAYER_ID = int(os.environ.get("LAYER_ID", "0"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "64"))

torch.set_grad_enabled(False)

print(f"MODEL_ID={MODEL_ID}")
print(f"LAYER_ID={LAYER_ID}")
print(f"SEQ_LEN={SEQ_LEN}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

model.eval()
device = next(model.parameters()).device
dtype = next(model.parameters()).dtype

cfg = model.config
layer = model.model.layers[LAYER_ID]
attn = layer.self_attn

prompt = "Explain KV cache quantization briefly. " * 20
enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
input_ids = enc["input_ids"].to(device)
T = input_ids.shape[1]
B = input_ids.shape[0]

print(f"Actual T={T}")
print(f"dtype={dtype}")

# For layer 0 only, this is the correct layer input.
hidden = model.model.embed_tokens(input_ids)
hidden_norm = layer.input_layernorm(hidden)

position_ids = torch.arange(T, device=device).unsqueeze(0)
cache_position = torch.arange(T, device=device)

# Build causal mask.
min_val = torch.finfo(hidden_norm.dtype).min
causal = torch.full((T, T), min_val, device=device, dtype=hidden_norm.dtype)
causal = torch.triu(causal, diagonal=1)
attention_mask = causal.unsqueeze(0).unsqueeze(0)

# HF original attention.
pos_emb = model.model.rotary_emb(hidden_norm, position_ids)

orig_out = attn(
    hidden_norm,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=cache_position,
    position_embeddings=pos_emb,
)[0]

# Manual RoPE-aware attention.
num_heads = cfg.num_attention_heads
num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
head_dim = cfg.hidden_size // num_heads
num_key_value_groups = num_heads // num_kv_heads

q = attn.q_proj(hidden_norm)
k = attn.k_proj(hidden_norm)
v = attn.v_proj(hidden_norm)

q = q.view(B, T, num_heads, head_dim).transpose(1, 2)      # (B, Hq, T, D)
k = k.view(B, T, num_kv_heads, head_dim).transpose(1, 2)   # (B, Hkv, T, D)
v = v.view(B, T, num_kv_heads, head_dim).transpose(1, 2)   # (B, Hkv, T, D)

cos, sin = pos_emb
q, k = apply_rotary_pos_emb(q, k, cos, sin)

k = repeat_kv(k, num_key_value_groups)
v = repeat_kv(v, num_key_value_groups)

scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
scores = scores + attention_mask
attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
manual = torch.matmul(attn_w, v)

manual = manual.transpose(1, 2).contiguous().reshape(B, T, cfg.hidden_size)
manual_out = attn.o_proj(manual)

diff = manual_out - orig_out
cosine = F.cosine_similarity(
    manual_out.reshape(1, -1).float(),
    orig_out.reshape(1, -1).float(),
).item()
rel_l2 = diff.float().norm().item() / (orig_out.float().norm().item() + 1e-8)
max_abs = diff.float().abs().max().item()
mean_abs = diff.float().abs().mean().item()

print("=" * 80)
print("HF original attention vs manual RoPE attention")
print(f"shape_orig={tuple(orig_out.shape)}")
print(f"shape_manual={tuple(manual_out.shape)}")
print(f"cosine={cosine:.8f}")
print(f"rel_l2={rel_l2:.8f}")
print(f"max_abs={max_abs:.8f}")
print(f"mean_abs={mean_abs:.8f}")
print("=" * 80)

os.makedirs("results", exist_ok=True)
with open(f"results/attention_rope_manual_layer{LAYER_ID}.txt", "w") as f:
    f.write(f"MODEL_ID={MODEL_ID}\n")
    f.write(f"LAYER_ID={LAYER_ID}\n")
    f.write(f"SEQ_LEN={T}\n")
    f.write(f"cosine={cosine:.8f}\n")
    f.write(f"rel_l2={rel_l2:.8f}\n")
    f.write(f"max_abs={max_abs:.8f}\n")
    f.write(f"mean_abs={mean_abs:.8f}\n")
