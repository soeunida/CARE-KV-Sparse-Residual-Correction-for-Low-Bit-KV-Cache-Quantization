import os, math, torch
from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import CacheConfig, patch_llama_model

MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODE = os.environ.get("MODE", "fp16")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))

texts = [
    ("KV cache quantization reduces memory usage during autoregressive decoding. "
     "A low-bit cache can reduce bandwidth, but quantization error may damage attention outputs. ") * 40,
    ("Large language models use key and value caches to avoid recomputing previous tokens. "
     "Compressing the cache is useful for long context inference and high batch serving. ") * 40,
]

def div(n, limit):
    limit = min(n, limit)
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None or tok.pad_token_id < 0:
    tok.pad_token_id = tok.eos_token_id or 0

model = LlamaForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.config.use_cache = False
model.generation_config.use_cache = False
model.generation_config.pad_token_id = tok.pad_token_id

if MODE != "fp16":
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    care_cfg = CacheConfig(
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        head_dim=head_dim,
        base_bits=int(os.environ.get("BASE_BITS", "4")),
        group_size=div(head_dim, 64),
        k_channel_group=div(head_dim, 32),
        store_budget_ratio=float(os.environ.get("STORE_BUDGET", "0.10")),
        read_budget_ratio=float(os.environ.get("READ_BUDGET", "0.03")),
    )
    care_cfg.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    model = patch_llama_model(model, care_cfg)

model.eval()
total_loss, total_tokens = 0.0, 0
device = next(model.parameters()).device

with torch.no_grad():
    for text in texts:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
        input_ids = enc["input_ids"].to(device)
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        n = input_ids.numel() - 1
        total_loss += out.loss.item() * n
        total_tokens += n

ppl = math.exp(total_loss / total_tokens)

print("=" * 60)
print(f"MODEL={MODEL_ID}")
print(f"MODE={MODE}")
print(f"BASE_BITS={os.environ.get('BASE_BITS', 'fp16')}")
print(f"SEQ_LEN={SEQ_LEN}")
print(f"TOKENS={total_tokens}")
print(f"PPL={ppl:.4f}")
print("=" * 60)
