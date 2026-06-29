import os, torch
from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import CacheConfig, patch_llama_model

MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
MODE = os.environ.get("MODE", "fp16")
NUM_DISTRACTORS = int(os.environ.get("NUM_DISTRACTORS", "200"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048"))
SECRET = "ZXQ-7391"

def div(n, limit):
    limit = min(n, limit)
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1

docs = []
for i in range(NUM_DISTRACTORS):
    docs.append(f"Document {i}: The access code is AAA-{1000+i}.")
docs.insert(NUM_DISTRACTORS // 2, f"Important document: The correct access code is {SECRET}.")
prompt = "\n".join(docs) + "\n\nQuestion: What is the correct access code? Answer:"

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
inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        use_cache=False,
        pad_token_id=tok.pad_token_id,
    )

txt = tok.decode(out[0], skip_special_tokens=True)
ans = txt[len(prompt):].strip()
hit = SECRET in ans

print("=" * 60)
print(f"MODE={MODE}")
print(f"BASE_BITS={os.environ.get('BASE_BITS', 'fp16')}")
print(f"NUM_DISTRACTORS={NUM_DISTRACTORS}")
print(f"SECRET={SECRET}")
print(f"ANSWER={ans}")
print(f"HIT={hit}")
print("=" * 60)
