import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

model_id = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print(f"Loading FP16 baseline: {model_id}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = LlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)

if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0

model.config.use_cache = False
model.generation_config.use_cache = False
model.generation_config.pad_token_id = tokenizer.pad_token_id

messages = [
    {"role": "user", "content": "Explain KV cache quantization briefly."}
]

if hasattr(tokenizer, "apply_chat_template"):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
else:
    prompt = "Explain KV cache quantization briefly."

inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=96,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
    )

print("\nOutput:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
