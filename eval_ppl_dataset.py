"""
eval_ppl_dataset.py
--------------------
Standard dataset PPL evaluation for CARE-KV.

Defaults to WikiText-2 (`wikitext-2-raw-v1`, test split).  C4 supported by
passing CAREKV_DATASET_NAME=c4 (or `--dataset c4`).

PPL is computed via non-overlapping windowed evaluation:
  - tokenize the dataset's text column
  - take the first NUM_SAMPLES × SEQ_LEN tokens
  - for each window, compute next-token cross-entropy via model.forward(labels=...)
  - PPL = exp(sum(loss × num_tokens_in_window) / total_tokens)

Environment variables (all optional; CLI overrides):
  MODEL_ID, DATASET_NAME, DATASET_CONFIG, DATASET_SPLIT, SEQ_LEN, NUM_SAMPLES,
  STRIDE (unused for now; reserved), MODE (label), BASE_BITS,
  CAREKV_PREFILL_MODE, CAREKV_PREFILL_RESIDUAL_KIND, CAREKV_ROUTE_POLICY,
  CAREKV_SCORE_NORMALIZE, CAREKV_CORRECTION_IMPL, CAREKV_PACKED_BASE,
  CAREKV_SCALE_QUANT,
  STORE_ABS_K, STORE_ABS_V, READ_ABS_K, READ_ABS_V, CAREKV_DEBUG_STATS.

Output (one row appended to --out-csv):
  model, dataset, config, split, mode, seq_len, num_samples, total_tokens,
  ppl, seconds, K_reads, V_reads, peak_gpu_mem_MB
"""

from __future__ import annotations
import argparse, csv, math, os, sys, time
from typing import List, Optional
import torch

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats, apply_carekv_env_overrides,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_text(dataset: str, config: str, split: str) -> List[str]:
    """Returns a list of non-empty text rows from the chosen dataset."""
    from datasets import load_dataset
    if dataset == "wikitext":
        ds = load_dataset("wikitext", config or "wikitext-2-raw-v1", split=split)
        texts = [r for r in ds["text"] if r.strip()]
    elif dataset == "c4":
        # C4 is huge — stream just enough.
        ds = load_dataset("allenai/c4", config or "en", split=split, streaming=True)
        texts = []
        for i, ex in enumerate(ds):
            if ex.get("text", "").strip():
                texts.append(ex["text"])
            if len(texts) >= 2000:
                break
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    return texts


def _build_eval_chunks(tok, texts: List[str], seq_len: int, num_samples: int):
    """Concatenate text, tokenize, split into N non-overlapping windows of
    seq_len tokens each."""
    full = "\n\n".join(texts)
    ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    needed = seq_len * num_samples
    if ids.numel() < needed:
        raise RuntimeError(f"dataset too short: {ids.numel()} tokens, need {needed}")
    chunks = ids[: needed].view(num_samples, seq_len)
    return chunks


def _make_model(model_id: str, mode: str, base_bits: int):
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    m.config.use_cache = False
    if mode == "fp16":
        m.eval(); return m
    # CARE-KV-patched mode
    cfg = m.config; hd = cfg.hidden_size // cfg.num_attention_heads
    kw = dict(
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd, base_bits=base_bits,
        group_size=32, k_channel_group=32, page_size=16, max_pages=256,
        v_token_block=4, sketch_dim=16,
        # Defaults match the paper-best config; env overrides take precedence.
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
        packed_base=True, scale_quant="int8",
        route_policy="joint", correction_impl="cached", budget_policy="uniform",
    )
    apply_carekv_env_overrides(kw)
    cc = CacheConfig(**kw)
    m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()
    return m


def _eval_one(model, chunks, mode_label: str):
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    reset_debug_stats()

    N, T = chunks.shape
    total_loss = 0.0
    total_tokens = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N):
            input_ids = chunks[i:i+1].to(DEVICE)
            out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            # HF CE loss is averaged over the (T-1) shifted positions
            n = T - 1
            total_loss += float(out.loss.item()) * n
            total_tokens += n
            # Reset CARE-KV per-sequence cache between windows so we don't
            # accumulate state across unrelated text chunks.
            if hasattr(model, "modules"):
                for sub in model.modules():
                    if hasattr(sub, "reset_cache") and hasattr(sub, "_caches"):
                        sub.reset_cache()
    if DEVICE == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ppl = math.exp(total_loss / total_tokens)
    stats = get_debug_stats()
    peak = (torch.cuda.max_memory_allocated()/1e6) if DEVICE == "cuda" else 0.0
    return dict(
        mode=mode_label,
        ppl=round(ppl, 6),
        total_tokens=total_tokens,
        seconds=round(dt, 2),
        K_reads=stats.get("k_slots_read", 0),
        V_reads=stats.get("v_slots_read", 0),
        peak_gpu_mem_MB=round(peak, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=os.environ.get(
        "MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--dataset", default=os.environ.get("DATASET_NAME", "wikitext"))
    ap.add_argument("--dataset-config", default=os.environ.get(
        "DATASET_CONFIG", "wikitext-2-raw-v1"))
    ap.add_argument("--dataset-split", default=os.environ.get("DATASET_SPLIT", "test"))
    ap.add_argument("--seq-len", type=int, default=int(os.environ.get("SEQ_LEN", "128")))
    ap.add_argument("--num-samples", type=int, default=int(
        os.environ.get("NUM_SAMPLES", "8")))
    ap.add_argument("--mode-label", default=os.environ.get("MODE", "auto"))
    ap.add_argument("--mode", choices=["fp16","base_quant","carekv_stored"],
                    default=None,
                    help="What kind of run.  fp16 = no CARE-KV patch; "
                         "base_quant or carekv_stored require base-bits.")
    ap.add_argument("--base-bits", type=int,
                    default=int(os.environ.get("BASE_BITS", "3")))
    ap.add_argument("--append-csv", required=True)
    args = ap.parse_args()

    # Infer --mode from MODE env if not provided
    if args.mode is None:
        if args.mode_label == "fp16":
            args.mode = "fp16"
        else:
            pm = os.environ.get("CAREKV_PREFILL_MODE", "carekv_stored")
            args.mode = "base_quant" if pm == "base_quant" else "carekv_stored"

    print(f"[ppl-dataset] dataset={args.dataset}/{args.dataset_config} split={args.dataset_split}",
          flush=True)
    print(f"[ppl-dataset] SEQ_LEN={args.seq_len} NUM_SAMPLES={args.num_samples} mode={args.mode}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None or tok.pad_token_id < 0:
        tok.pad_token_id = tok.eos_token_id or 0

    texts = _load_text(args.dataset, args.dataset_config, args.dataset_split)
    chunks = _build_eval_chunks(tok, texts, args.seq_len, args.num_samples)
    print(f"[ppl-dataset] built {chunks.shape[0]} windows × {chunks.shape[1]} tokens",
          flush=True)

    # Ensure carekv-related env vars get applied through apply_carekv_env_overrides
    # by setting CAREKV_PREFILL_MODE based on --mode argument
    if args.mode == "fp16":
        os.environ["CAREKV_PREFILL_MODE"] = "fp"
    elif args.mode == "base_quant":
        os.environ["CAREKV_PREFILL_MODE"] = "base_quant"
    else:
        os.environ["CAREKV_PREFILL_MODE"] = "carekv_stored"

    if args.mode_label == "auto":
        if args.mode == "fp16":
            args.mode_label = "fp16"
        elif args.mode == "base_quant":
            args.mode_label = f"base_quant_int{args.base_bits}"
        else:
            args.mode_label = f"carekv_stored_int{args.base_bits}"

    model = _make_model(args.model_id, args.mode, args.base_bits)
    res = _eval_one(model, chunks, args.mode_label)
    del model
    if DEVICE == "cuda": torch.cuda.empty_cache()

    row = dict(
        model=args.model_id,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
        base_bits=args.base_bits,
        prefill_mode=os.environ.get("CAREKV_PREFILL_MODE", "?"),
        route_policy=os.environ.get("CAREKV_ROUTE_POLICY", "joint"),
        correction_impl=os.environ.get("CAREKV_CORRECTION_IMPL", "cached"),
        packed_base=os.environ.get("CAREKV_PACKED_BASE", "1"),
        scale_quant=os.environ.get("CAREKV_SCALE_QUANT", "int8"),
        store_abs_k=os.environ.get("STORE_ABS_K", "-"),
        store_abs_v=os.environ.get("STORE_ABS_V", "-"),
        read_abs_k=os.environ.get("READ_ABS_K", "-"),
        read_abs_v=os.environ.get("READ_ABS_V", "-"),
        **res,
    )

    write_header = not os.path.exists(args.append_csv) or os.path.getsize(args.append_csv) == 0
    os.makedirs(os.path.dirname(args.append_csv) or ".", exist_ok=True)
    with open(args.append_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"[ppl-dataset] mode={args.mode_label}  PPL={res['ppl']:.4f}  "
          f"tokens={res['total_tokens']}  {res['seconds']:.1f}s  "
          f"K_reads={res['K_reads']}  V_reads={res['V_reads']}  "
          f"peak={res['peak_gpu_mem_MB']}MB", flush=True)


if __name__ == "__main__":
    main()
