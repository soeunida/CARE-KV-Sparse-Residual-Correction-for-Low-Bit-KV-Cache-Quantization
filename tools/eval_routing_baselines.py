"""tools/eval_routing_baselines.py

Phase M — routing baseline ablation for CARE-KV.

Compares 6 residual-routing baselines under the **same** store/read budget
to isolate whether CARE-KV's joint+normalized score actually selects useful
residual slots vs simpler alternatives:

  1. base_quant       — no residual correction (RK=RV=0)
  2. random           — uniform random per-candidate scores
  3. magnitude_only   — score = |q·R_K| (K) / ||R_V|| (V)  (no attention)
  4. attention_only   — score = page_attn_mass (K) / blk_attn_mass (V)  (no magnitude)
  5. carekv_score     — current paper-best score (kind=both, joint, normalize=1)
  6. oracle_proxy     — score = magnitude × attention (no structural prior /
                        sensitivity). Diagnostic upper-bound, not a deployable
                        method.

All six use the SAME store budget (SK=2, SV=4) and SAME read budget
(RK=2, RV=2) — only the per-candidate score formula changes. Paper-best
CARE-KV (carekv_score) is unchanged at the config level.

Defaults: synthetic 254-token prompt (~12 min wall-clock for all 6 cells).
Pass --dataset wikitext for the full WikiText-2 path (much slower).

Outputs:
  results/paper_eval_20260529_015053/ablations/routing_baseline_ablation.csv
  ./<--out-dir>/<summary png if --figure>
"""
from __future__ import annotations
import argparse, csv, os, sys, time

import torch

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats, estimate_memory_bytes,
)
from CARE_KV.care_kv.cache import apply_carekv_env_overrides

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paper-best config (do NOT change). The only thing we vary per baseline
# is CAREKV_BASELINE_SCORE (+ RK=RV=0 for base_quant).
PAPER_BEST_ENV = dict(
    CAREKV_PACKED_BASE="1",
    CAREKV_SCALE_QUANT="int8",
    CAREKV_PREFILL_MODE="carekv_stored",
    CAREKV_PREFILL_RESIDUAL_KIND="both",
    CAREKV_ROUTE_POLICY="joint",
    CAREKV_SCORE_NORMALIZE="1",
    CAREKV_CORRECTION_IMPL="cached",
    CAREKV_BUDGET_POLICY="uniform",
    CAREKV_DEBUG_STATS="1",
    CAREKV_STORE_BUDGET_MODE="absolute",
    CAREKV_READ_BUDGET_MODE="absolute",
    CAREKV_STORE_ABS_K="2",
    CAREKV_STORE_ABS_V="4",
    CAREKV_READ_ABS_K="2",
    CAREKV_READ_ABS_V="2",
)


# ─────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────

# Each baseline: (label, env-override dict)
BASELINES = [
    ("base_quant",      {"CAREKV_PREFILL_MODE": "base_quant",
                          "CAREKV_READ_ABS_K": "0", "CAREKV_READ_ABS_V": "0",
                          "CAREKV_BASELINE_SCORE": "carekv"}),
    ("random",          {"CAREKV_BASELINE_SCORE": "random"}),
    ("magnitude_only",  {"CAREKV_BASELINE_SCORE": "magnitude_only"}),
    ("attention_only",  {"CAREKV_BASELINE_SCORE": "attention_only"}),
    ("carekv_score",    {"CAREKV_BASELINE_SCORE": "carekv"}),
    ("oracle_proxy",    {"CAREKV_BASELINE_SCORE": "oracle_proxy"}),
]


# ─────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────

SYNTHETIC_PROMPT = (
    "The CARE-KV project investigates low-bit KV cache quantization for "
    "transformer attention. We focus on int3 base quantization with sparse "
    "residual correction. The router selects residual slots that have the "
    "highest expected output-error contribution. We compare against random "
    "selection, magnitude-only ranking, and attention-only ranking to "
    "establish that the joint score actually picks useful residuals. The "
    "experiment runs on TinyLlama-1.1B and reports perplexity together with "
    "read counts so we can verify the router fires consistently across "
    "different routing baselines, ensuring the comparison is fair across "
    "all candidate scoring policies considered in this ablation study. "
) * 4  # roughly 254-token prompt before tokenization


def _build_model(env_overrides: dict, base_bits: int = 3):
    """Load TinyLlama + patch with CARE-KV using paper-best + per-baseline env."""
    full_env = {**PAPER_BEST_ENV, **env_overrides,
                "CAREKV_BASE_BITS": str(base_bits)}
    for k, v in full_env.items():
        os.environ[k] = v
    # Force re-import-free env application: patch_llama_model reads env
    # via apply_carekv_env_overrides at construction.
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    m.config.use_cache = False
    cfg = m.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    kw = dict(
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd, base_bits=base_bits,
        group_size=32, k_channel_group=32, page_size=16, max_pages=512,
        v_token_block=4, sketch_dim=16,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
    )
    # Honor every CAREKV_* env (incl. STORE/READ_ABS_*, baseline_score,
    # route_policy, scale_quant, packed_base, correction_impl, etc.).
    apply_carekv_env_overrides(kw)
    cc = CacheConfig(**kw)
    m = patch_llama_model(m, cc)
    reset_all_caches(m)
    m.eval()
    return m


def _ppl_synthetic(m, tok, seq_len: int):
    """Compute PPL on a single synthetic prompt (fixed seed)."""
    enc = tok(SYNTHETIC_PROMPT, return_tensors="pt",
              truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"].to(DEVICE)
    T = int(input_ids.shape[1])
    with torch.no_grad():
        out = m(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = float(out.loss.item())
    ppl = float(torch.exp(torch.tensor(loss)).item())
    return ppl, T


def _ppl_wikitext(m, tok, seq_len: int, num_samples: int):
    """Compute PPL on WikiText-2 windowed log-loss."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    # Build N non-overlapping windows of seq_len tokens
    windows = []
    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len
        if end <= ids.numel():
            windows.append(ids[start:end])
    if not windows:
        raise RuntimeError("not enough tokens for any window")
    total_loss = 0.0
    total_tokens = 0
    for w in windows:
        ids_w = w.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = m(input_ids=ids_w, labels=ids_w, use_cache=False)
        n = ids_w.numel() - 1  # CE excludes the last shifted token
        total_loss += float(out.loss.item()) * n
        total_tokens += n
    mean_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return ppl, total_tokens


# ─────────────────────────────────────────────
# Per-cell runner
# ─────────────────────────────────────────────

def run_cell(label: str, env_overrides: dict,
             dataset: str, seq_len: int, num_samples: int,
             base_bits: int = 3):
    print(f"--- baseline: {label}  env: {env_overrides} ---", flush=True)
    reset_debug_stats()
    t0 = time.perf_counter()
    m = _build_model(env_overrides, base_bits=base_bits)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    if dataset == "synthetic":
        ppl, total_tokens = _ppl_synthetic(m, tok, seq_len)
    else:
        ppl, total_tokens = _ppl_wikitext(m, tok, seq_len, num_samples)
    dt = time.perf_counter() - t0
    stats = get_debug_stats()
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return dict(
        label=label,
        baseline_score=env_overrides.get("CAREKV_BASELINE_SCORE", "carekv"),
        prefill_mode=env_overrides.get("CAREKV_PREFILL_MODE",
                                        PAPER_BEST_ENV["CAREKV_PREFILL_MODE"]),
        read_abs_k=env_overrides.get("CAREKV_READ_ABS_K",
                                      PAPER_BEST_ENV["CAREKV_READ_ABS_K"]),
        read_abs_v=env_overrides.get("CAREKV_READ_ABS_V",
                                      PAPER_BEST_ENV["CAREKV_READ_ABS_V"]),
        store_abs_k=PAPER_BEST_ENV["CAREKV_STORE_ABS_K"],
        store_abs_v=PAPER_BEST_ENV["CAREKV_STORE_ABS_V"],
        route_policy=PAPER_BEST_ENV["CAREKV_ROUTE_POLICY"],
        ppl=round(ppl, 4),
        total_tokens=int(total_tokens),
        seconds=round(dt, 1),
        K_reads=int(stats.get("k_slots_read", 0)),
        V_reads=int(stats.get("v_slots_read", 0)),
        K_stored=int(stats.get("k_slots_stored", 0)),
        V_stored=int(stats.get("v_slots_stored", 0)),
        mean_dO_K=float(stats.get("mean_dO_K", 0.0) or 0.0),
        mean_dO_V=float(stats.get("mean_dO_V", 0.0) or 0.0),
        base_bits=base_bits,
        dataset=dataset,
        seq_len=seq_len,
        num_samples=num_samples,
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "wikitext"])
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--baselines", default=None,
                    help="comma-separated subset of baseline labels")
    args = ap.parse_args()

    wanted = set(args.baselines.split(",")) if args.baselines else None
    rows = []
    for label, env in BASELINES:
        if wanted is not None and label not in wanted:
            continue
        try:
            r = run_cell(label, env, args.dataset, args.seq_len,
                         args.num_samples, args.base_bits)
            rows.append(r)
            print(f"[routing-bl] {label:18s} PPL={r['ppl']:.4f}  "
                  f"K_reads={r['K_reads']}  V_reads={r['V_reads']}  "
                  f"({r['seconds']:.1f}s)", flush=True)
        except Exception as e:
            print(f"[routing-bl] {label} ERROR: {type(e).__name__}: {e}",
                  flush=True)
            rows.append(dict(label=label, error=str(e),
                             baseline_score=env.get("CAREKV_BASELINE_SCORE", "")))

    if rows:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"wrote {len(rows)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
