"""tools/eval_adaptive_read_budget.py

Focused read-budget experiment for the adaptive_score mode.

Compares two regimes under the paper-best store budget
(SK=2, SV=4) on the same synthetic 254-token prompt at SL=64:

  Group 1 — fixed read budget (current paper-best mode):
    RK=RV=1, 2, 3, 4

  Group 2 — adaptive_score with MAX RK=RV=4 and varying relative threshold:
    CAREKV_READ_RELATIVE_THRESHOLD in {0.00, 0.05, 0.10, 0.20, 0.30}
    CAREKV_READ_ABSOLUTE_THRESHOLD = 0.0  (not used in this pass)

Per-cell metrics:
  PPL, K_reads, V_reads, effective_RK_mean, effective_RV_mean,
  mean |ΔO_K|, mean |ΔO_V|, runtime seconds, the per-kind threshold-skip
  counters (skipped_*_by_relative_threshold, skipped_*_by_absolute_threshold).

Output:
  results/paper_eval_20260529_015053/ablations/adaptive_read_budget.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats,
)
from CARE_KV.care_kv.cache import apply_carekv_env_overrides

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paper-best (do not change). Per-cell env adds the read-budget knobs.
PAPER_BEST_ENV = dict(
    CAREKV_PACKED_BASE="1",
    CAREKV_SCALE_QUANT="int8",
    CAREKV_PREFILL_MODE="carekv_stored",
    CAREKV_PREFILL_RESIDUAL_KIND="both",
    CAREKV_ROUTE_POLICY="joint",
    CAREKV_SCORE_NORMALIZE="1",
    CAREKV_CORRECTION_IMPL="cached",
    CAREKV_BUDGET_POLICY="uniform",
    CAREKV_STORE_BUDGET_MODE="absolute",
    CAREKV_STORE_ABS_K="2",
    CAREKV_STORE_ABS_V="4",
    CAREKV_DEBUG_STATS="1",
)

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
) * 4


def _cell_fixed(rk: int, rv: int) -> Tuple[str, Dict[str, str]]:
    label = f"fixed_RK{rk}_RV{rv}"
    env = dict(
        CAREKV_READ_BUDGET_MODE="absolute",
        CAREKV_READ_ABS_K=str(rk), CAREKV_READ_ABS_V=str(rv),
        # Clear adaptive knobs
        CAREKV_READ_RELATIVE_THRESHOLD="0.0",
        CAREKV_READ_ABSOLUTE_THRESHOLD="0.0",
        CAREKV_READ_MIN_KEEP="0",
    )
    return label, env


def _cell_adaptive(max_rk: int, max_rv: int, rel_thr: float,
                   abs_thr: float = 0.0, min_keep: int = 0) -> Tuple[str, Dict[str, str]]:
    label = f"adaptive_maxRK{max_rk}_maxRV{max_rv}_rel{rel_thr:.2f}"
    env = dict(
        CAREKV_READ_BUDGET_MODE="adaptive_score",
        CAREKV_READ_ABS_K=str(max_rk), CAREKV_READ_ABS_V=str(max_rv),
        CAREKV_READ_RELATIVE_THRESHOLD=str(rel_thr),
        CAREKV_READ_ABSOLUTE_THRESHOLD=str(abs_thr),
        CAREKV_READ_MIN_KEEP=str(min_keep),
    )
    return label, env


def _cell_base_quant() -> Tuple[str, Dict[str, str]]:
    """Base_quant reference: carekv_stored prefill is skipped via prefill_mode."""
    env = dict(
        CAREKV_PREFILL_MODE="base_quant",
        CAREKV_READ_BUDGET_MODE="absolute",
        CAREKV_READ_ABS_K="0", CAREKV_READ_ABS_V="0",
        CAREKV_READ_RELATIVE_THRESHOLD="0.0",
        CAREKV_READ_ABSOLUTE_THRESHOLD="0.0",
        CAREKV_READ_MIN_KEEP="0",
    )
    return "base_quant", env


def build_cells(preset: str = "synthetic_full") -> List[Tuple[str, Dict[str, str]]]:
    """Cell list selector.

    preset="synthetic_full"  — original 9-cell sweep (4 fixed RK=RV={1,2,3,4}
                               + 5 adaptive max RK=RV=4 rel={0.00..0.30}).
    preset="wikitext2_n4"    — 7-cell sweep for WT-2 N=4 SL=128 confirmation
                               (base_quant + fixed RK=RV=2 + fixed RK=RV=4 +
                                4 adaptive max RK=RV=4 rel={0.05,0.10,0.20,0.30}).
    """
    if preset == "wikitext2_n4":
        cells = [_cell_base_quant(),
                 _cell_fixed(2, 2),
                 _cell_fixed(4, 4)]
        for rel in (0.05, 0.10, 0.20, 0.30):
            cells.append(_cell_adaptive(4, 4, rel))
        return cells
    # default
    cells = []
    for r in (1, 2, 3, 4):
        cells.append(_cell_fixed(r, r))
    for rel in (0.00, 0.05, 0.10, 0.20, 0.30):
        cells.append(_cell_adaptive(4, 4, rel))
    return cells


def _build_model(env_overrides: Dict[str, str], base_bits: int = 3):
    full_env = {**PAPER_BEST_ENV, **env_overrides,
                "CAREKV_BASE_BITS": str(base_bits)}
    for k, v in full_env.items():
        os.environ[k] = v
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
    apply_carekv_env_overrides(kw)
    cc = CacheConfig(**kw)
    m = patch_llama_model(m, cc)
    reset_all_caches(m)
    m.eval()
    return m, cc


def _ppl_synthetic(m, tok, seq_len: int):
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
    """N non-overlapping windowed log-loss PPL on WikiText-2 test."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tok(text, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    windows = []
    for i in range(num_samples):
        start = i * seq_len
        end = start + seq_len
        if end <= ids.numel():
            windows.append(ids[start:end])
    if not windows:
        raise RuntimeError("not enough WT-2 tokens for any window")
    total_loss = 0.0
    total_tokens = 0
    for w in windows:
        ids_w = w.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = m(input_ids=ids_w, labels=ids_w, use_cache=False)
        n = ids_w.numel() - 1
        total_loss += float(out.loss.item()) * n
        total_tokens += n
    mean_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return ppl, total_tokens


def run_cell(label: str, env: Dict[str, str], seq_len: int, base_bits: int = 3,
             dataset: str = "synthetic", num_samples: int = 4):
    t0 = time.perf_counter()
    reset_debug_stats()
    m, cc = _build_model(env, base_bits=base_bits)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    if dataset == "wikitext":
        ppl, total_tokens = _ppl_wikitext(m, tok, seq_len, num_samples)
    else:
        ppl, total_tokens = _ppl_synthetic(m, tok, seq_len)
    dt = time.perf_counter() - t0
    stats = get_debug_stats()
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    n_calls = max(1, int(stats.get("router_n_route_calls", 1)))
    eff_rk_mean = float(stats.get("router_effective_RK_sum", 0)) / n_calls
    eff_rv_mean = float(stats.get("router_effective_RV_sum", 0)) / n_calls
    req_rk_mean = float(stats.get("router_requested_RK", 0)) / n_calls
    req_rv_mean = float(stats.get("router_requested_RV", 0)) / n_calls

    return dict(
        label=label,
        mode=env.get("CAREKV_READ_BUDGET_MODE", "absolute"),
        max_RK=int(env.get("CAREKV_READ_ABS_K", "0")),
        max_RV=int(env.get("CAREKV_READ_ABS_V", "0")),
        rel_threshold=float(env.get("CAREKV_READ_RELATIVE_THRESHOLD", "0.0")),
        abs_threshold=float(env.get("CAREKV_READ_ABSOLUTE_THRESHOLD", "0.0")),
        min_keep=int(env.get("CAREKV_READ_MIN_KEEP", "0")),
        store_abs_k=PAPER_BEST_ENV["CAREKV_STORE_ABS_K"],
        store_abs_v=PAPER_BEST_ENV["CAREKV_STORE_ABS_V"],
        ppl=round(ppl, 4),
        total_tokens=int(total_tokens),
        seconds=round(dt, 1),
        K_reads=int(stats.get("k_slots_read", 0)),
        V_reads=int(stats.get("v_slots_read", 0)),
        requested_RK_mean=round(req_rk_mean, 3),
        requested_RV_mean=round(req_rv_mean, 3),
        effective_RK_mean=round(eff_rk_mean, 3),
        effective_RV_mean=round(eff_rv_mean, 3),
        skipped_K_by_relative_threshold=int(stats.get(
            "router_skipped_K_by_relative_threshold", 0)),
        skipped_K_by_absolute_threshold=int(stats.get(
            "router_skipped_K_by_absolute_threshold", 0)),
        skipped_V_by_relative_threshold=int(stats.get(
            "router_skipped_V_by_relative_threshold", 0)),
        skipped_V_by_absolute_threshold=int(stats.get(
            "router_skipped_V_by_absolute_threshold", 0)),
        n_route_calls=int(stats.get("router_n_route_calls", 0)),
        mean_dO_K=float(stats.get("mean_dO_K", 0.0) or 0.0),
        mean_dO_V=float(stats.get("mean_dO_V", 0.0) or 0.0),
        base_bits=base_bits,
        seq_len=seq_len,
        dataset=dataset,
        num_samples=num_samples,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "wikitext"])
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--preset", default="synthetic_full",
                    choices=["synthetic_full", "wikitext2_n4"])
    args = ap.parse_args()

    rows = []
    for label, env in build_cells(args.preset):
        try:
            r = run_cell(label, env, args.seq_len, args.base_bits,
                          dataset=args.dataset, num_samples=args.num_samples)
            rows.append(r)
            print(f"[adaptive] {label:42s} PPL={r['ppl']:.4f}  "
                  f"req=({r['requested_RK_mean']:.2f},{r['requested_RV_mean']:.2f}) "
                  f"eff=({r['effective_RK_mean']:.2f},{r['effective_RV_mean']:.2f})  "
                  f"K={r['K_reads']} V={r['V_reads']}  ({r['seconds']:.1f}s)",
                  flush=True)
        except Exception as e:
            print(f"[adaptive] {label} ERROR: {type(e).__name__}: {e}", flush=True)
            rows.append(dict(label=label, error=str(e)))

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
