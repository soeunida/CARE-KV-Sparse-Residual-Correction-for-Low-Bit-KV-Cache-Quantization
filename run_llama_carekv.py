import os
import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from CARE_KV.care_kv import CacheConfig

try:
    from CARE_KV.care_kv import patch_llama_model
except ImportError:
    patch_llama_model = None

try:
    from CARE_KV.care_kv import reset_all_caches
except ImportError:
    reset_all_caches = None


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def largest_divisor_leq(n: int, limit: int) -> int:
    limit = min(n, limit)
    for d in range(limit, 0, -1):
        if n % d == 0:
            return d
    return 1


def build_prompt(tokenizer, raw_prompt: str) -> str:
    use_chat_template = str2bool(os.environ.get("USE_CHAT_TEMPLATE", "1"))

    if not use_chat_template:
        return raw_prompt

    messages = [
        {"role": "user", "content": raw_prompt}
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return raw_prompt


def main():
    model_id = os.environ.get(
        "MODEL_ID",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
    )

    mode = os.environ.get("MODE", "carekv").lower()
    use_carekv = mode not in {"fp16", "baseline", "none"}

    use_cache = str2bool(os.environ.get("USE_CACHE", "0"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "64"))

    do_sample = str2bool(os.environ.get("DO_SAMPLE", "1"))
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    top_p = float(os.environ.get("TOP_P", "0.9"))

    disable_eos = str2bool(os.environ.get("DISABLE_EOS", "0"))

    raw_prompt = os.environ.get(
        "PROMPT",
        "Explain KV cache quantization briefly.",
    )

    print(f"Loading model: {model_id}")
    print(f"MODE={mode}")
    print(f"USE_CACHE={use_cache}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # CARE-KV current smoke-test path should use use_cache=False.
    # use_cache=True requires HuggingFace past_key_values integration.
    model.config.use_cache = use_cache
    model.generation_config.use_cache = use_cache

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id >= 0:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    if use_carekv:
        if patch_llama_model is None:
            raise RuntimeError("patch_llama_model could not be imported.")

        cfg = model.config

        head_dim = cfg.hidden_size // cfg.num_attention_heads
        group_size = largest_divisor_leq(
            head_dim,
            int(os.environ.get("GROUP_SIZE_LIMIT", "64")),
        )
        k_channel_group = largest_divisor_leq(
            head_dim,
            int(os.environ.get("K_CHANNEL_GROUP_LIMIT", "32")),
        )

        base_bits = int(os.environ.get("BASE_BITS", "2"))
        store_budget_ratio = float(os.environ.get("STORE_BUDGET", "0.10"))
        read_budget_ratio = float(os.environ.get("READ_BUDGET", "0.03"))

        print(
            f"CARE-KV config: head_dim={head_dim}, "
            f"group_size={group_size}, "
            f"k_channel_group={k_channel_group}, "
            f"base_bits={base_bits}, "
            f"store_budget={store_budget_ratio}, "
            f"read_budget={read_budget_ratio}"
        )

        care_cfg = CacheConfig(
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            head_dim=head_dim,
            base_bits=base_bits,
            group_size=group_size,
            k_channel_group=k_channel_group,
            store_budget_ratio=store_budget_ratio,
            read_budget_ratio=read_budget_ratio,
        )

        care_cfg.num_kv_heads = getattr(
            cfg,
            "num_key_value_heads",
            cfg.num_attention_heads,
        )

        print(
            f"CARE-KV GQA config: "
            f"q_heads={cfg.num_attention_heads}, "
            f"kv_heads={care_cfg.num_kv_heads}"
        )

        model = patch_llama_model(model, care_cfg)

        if reset_all_caches is not None:
            reset_all_caches(model)
    else:
        print("Running FP16/BF16 baseline without CARE-KV patch.")

    prompt = build_prompt(tokenizer, raw_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Running generation...")

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=use_cache,
        pad_token_id=tokenizer.pad_token_id,
    )

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    if disable_eos:
        gen_kwargs["eos_token_id"] = None

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nPrompt:")
    print(prompt)

    print("\nOutput:")
    print(decoded)


if __name__ == "__main__":
    main()
