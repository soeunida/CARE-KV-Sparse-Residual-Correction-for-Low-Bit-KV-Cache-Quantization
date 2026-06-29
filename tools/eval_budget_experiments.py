"""tools/eval_budget_experiments.py

Phase N — CARE-KV budget experiments.

Five sub-experiments, all on the same synthetic 254-token prompt at SL=64
(prototype-evaluation, diagnostic-only — small per-cell runtime so the
full 45-cell sweep finishes in ~90 min). Cross-experiment PPL comparisons
are meaningful because every cell uses the same forward-pass eval.

Subcommands (--experiment):

  A1   ratio budget grid          (store_ratio × read_ratio)
  A2   absolute budget grid       ((SK, SV, RK, RV) tuples)
  B    store budget sweep         (SK, SV varied; RK=RV=2 fixed)
  C    read budget sweep          (RK, RV varied; SK=2, SV=4 fixed)
  D    K/V budget balance         (K-only / V-only / K-heavy / V-heavy / balanced)
  all  runs A1, A2, B, C, D back-to-back

Paper-best CARE-KV config is used except for the budget knobs that each
experiment varies (and the kind override for D). NO change to the
paper-best method.

Outputs per experiment:
  results/paper_eval_20260529_015053/ablations/<experiment-name>.csv

Use `--out-dir <dir>` to redirect.
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time
from typing import Dict, List, Tuple

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

# Paper-best (do not change). The budget knobs below get overridden per cell.
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

# ─────────────────────────────────────────────
# Cell builders
# ─────────────────────────────────────────────

def cells_A1_ratio() -> List[Tuple[str, Dict[str, str]]]:
    """Ratio budget grid: (store_ratio, read_ratio) combinations."""
    cells = []
    for sr in [0.02, 0.05, 0.10, 0.20]:
        for rr in [0.01, 0.03, 0.10]:
            label = f"ratio_S{sr:.2f}_R{rr:.2f}"
            env = dict(
                CAREKV_STORE_BUDGET_MODE="ratio",
                CAREKV_READ_BUDGET_MODE="ratio",
                CAREKV_STORE_BUDGET=str(sr),
                CAREKV_READ_BUDGET=str(rr),
                # Clear any absolute leftovers
                CAREKV_STORE_ABS_K="0", CAREKV_STORE_ABS_V="0",
                CAREKV_READ_ABS_K="0",  CAREKV_READ_ABS_V="0",
            )
            cells.append((label, env))
    return cells


def _abs_env(sk: int, sv: int, rk: int, rv: int) -> Dict[str, str]:
    return dict(
        CAREKV_STORE_BUDGET_MODE="absolute",
        CAREKV_READ_BUDGET_MODE="absolute",
        CAREKV_STORE_ABS_K=str(sk), CAREKV_STORE_ABS_V=str(sv),
        CAREKV_READ_ABS_K=str(rk),  CAREKV_READ_ABS_V=str(rv),
        # Zero out ratio fallbacks
        CAREKV_STORE_BUDGET="0", CAREKV_READ_BUDGET="0",
    )


def cells_A2_absolute():
    """Absolute budget grid (matches user spec)."""
    grid = [
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (2, 2, 1, 1),
        (2, 4, 1, 1),
        (2, 4, 2, 2),
        (2, 4, 3, 3),
        (2, 4, 4, 4),
        (4, 4, 2, 2),
        (8, 4, 2, 2),
        (4, 8, 2, 2),
    ]
    return [(f"abs_SK{sk}_SV{sv}_RK{rk}_RV{rv}", _abs_env(sk, sv, rk, rv))
            for (sk, sv, rk, rv) in grid]


def cells_B_store():
    """Store budget sweep with RK=RV=2 fixed."""
    grid = [
        (0, 0), (1, 1), (1, 2), (2, 2),
        (2, 4), (4, 4), (8, 4), (4, 8),
    ]
    return [(f"store_SK{sk}_SV{sv}", _abs_env(sk, sv, 2, 2))
            for (sk, sv) in grid]


def cells_C_read():
    """Read budget sweep with SK=2, SV=4 fixed."""
    grid = [
        (0, 0), (1, 1), (1, 2), (2, 1),
        (2, 2), (2, 3), (3, 2), (3, 3), (4, 4),
    ]
    return [(f"read_RK{rk}_RV{rv}", _abs_env(2, 4, rk, rv))
            for (rk, rv) in grid]


def cells_D_balance():
    """K/V balance experiment, including kind=k / kind=v variants."""
    base = "kind=both"
    cells = [
        # K-heavy
        ("balance_K_heavy_2_1_2_1", {**_abs_env(2, 1, 2, 1),
                                       "CAREKV_PREFILL_RESIDUAL_KIND": "both"}),
        ("balance_K_heavy_2_2_3_1", {**_abs_env(2, 2, 3, 1),
                                       "CAREKV_PREFILL_RESIDUAL_KIND": "both"}),
        # V-heavy
        ("balance_V_heavy_1_4_1_2", {**_abs_env(1, 4, 1, 2),
                                       "CAREKV_PREFILL_RESIDUAL_KIND": "both"}),
        ("balance_V_heavy_2_4_1_3", {**_abs_env(2, 4, 1, 3),
                                       "CAREKV_PREFILL_RESIDUAL_KIND": "both"}),
        # Balanced (paper-best)
        ("balance_balanced_2_4_2_2", {**_abs_env(2, 4, 2, 2),
                                       "CAREKV_PREFILL_RESIDUAL_KIND": "both"}),
        # K-only / V-only
        ("balance_K_only", {**_abs_env(2, 0, 2, 0),
                              "CAREKV_PREFILL_RESIDUAL_KIND": "k"}),
        ("balance_V_only", {**_abs_env(0, 4, 0, 2),
                              "CAREKV_PREFILL_RESIDUAL_KIND": "v"}),
    ]
    return cells


# ─────────────────────────────────────────────
# Eval helpers
# ─────────────────────────────────────────────

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
    return m, cc, hd


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


def _mem_estimate(cc, hd, seq_len: int):
    try:
        b = estimate_memory_bytes(cc, num_tokens=seq_len)
        return float(b) / (1024.0 * 1024.0)
    except Exception:
        return -1.0


def _effective_budget_fields(cc, hd, seq_len: int, stats: Dict) -> Dict:
    """Candidate-cap-aware budget accounting.

    The store budget selects from per-page candidate POOLS whose size is fixed
    by the residual granularity, NOT by the budget knob:

        K candidate cap = head_dim     / k_channel_group
        V candidate cap = ceil(page_size / v_token_block)

    So `store_abs_k` above the K cap (and `store_abs_v` above the V cap) cannot
    select any new candidate — the *effective* budget saturates at the cap.
    This mirrors `utils.estimate_memory_bytes`, which already clamps stored
    slots to the same caps.  Reporting the requested vs effective budget makes
    the saturation explicit on every row.

    Recovered element counts are derived from the ACTUAL router reads
    (`k_slots_read` / `v_slots_read`): each read K slot reconstructs
    `page_size × k_channel_group` residual values; each read V slot
    reconstructs `v_token_block × head_dim` values.  These are throughput
    counts (summed over layers, query heads, pages, and decode steps), aligned
    with how `K_reads` / `V_reads` are already reported.
    """
    k_cap = hd // cc.k_channel_group
    v_cap = math.ceil(cc.page_size / cc.v_token_block)

    if cc.store_budget_mode == "absolute":
        req_sk = int(cc.store_abs_k)
        req_sv = int(cc.store_abs_v)
    else:
        req_sk = int(k_cap * (cc.store_budget_ratio or 0.0))
        req_sv = int(v_cap * (cc.store_budget_ratio or 0.0))

    eff_sk = max(0, min(req_sk, k_cap))
    eff_sv = max(0, min(req_sv, v_cap))
    denom = (k_cap + v_cap) or 1
    store_util = round((eff_sk + eff_sv) / denom, 4)
    # Was any requested budget thrown away by the cap?
    wasted = max(0, req_sk - k_cap) + max(0, req_sv - v_cap)

    k_reads = int(stats.get("k_slots_read", 0))
    v_reads = int(stats.get("v_slots_read", 0))
    rec_k = k_reads * cc.page_size * cc.k_channel_group
    rec_v = v_reads * cc.v_token_block * hd

    # Residual-only memory (mirrors the residual portion of
    # utils.estimate_memory_bytes; uses the *effective* per-page store slots).
    L, Hkv = cc.num_layers, cc.num_kv_heads
    n_pages = (seq_len + cc.page_size - 1) // cc.page_size
    k_slot_vals = cc.page_size * cc.k_channel_group
    v_slot_vals = cc.v_token_block * hd
    k_res_b = L * Hkv * n_pages * eff_sk * (k_slot_vals // 2)   # 4-bit packed
    v_res_b = L * Hkv * n_pages * eff_sv * (v_slot_vals // 2)
    res_scale_b = 2 * L * Hkv * n_pages * (eff_sk + eff_sv)     # one fp16/slot
    res_bytes = int(k_res_b + v_res_b + res_scale_b)

    return dict(
        k_channel_group=cc.k_channel_group,
        v_token_block=cc.v_token_block,
        page_size=cc.page_size,
        head_dim=hd,
        K_cand_cap=k_cap,
        V_cand_cap=v_cap,
        req_SK=req_sk,
        eff_SK=eff_sk,
        req_SV=req_sv,
        eff_SV=eff_sv,
        budget_wasted=wasted,
        store_util=store_util,
        recovered_K_elems=rec_k,
        recovered_V_elems=rec_v,
        residual_mem_bytes=res_bytes,
        residual_mem_MB=round(res_bytes / (1024.0 * 1024.0), 4),
    )


# ─────────────────────────────────────────────
# Per-cell runner
# ─────────────────────────────────────────────

def run_cell(label: str, env: Dict[str, str], seq_len: int, base_bits: int = 3):
    t0 = time.perf_counter()
    reset_debug_stats()
    m, cc, hd = _build_model(env, base_bits=base_bits)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    ppl, total_tokens = _ppl_synthetic(m, tok, seq_len)
    dt = time.perf_counter() - t0
    stats = get_debug_stats()
    mem_mb = _mem_estimate(cc, hd, seq_len)
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    row = dict(
        label=label,
        prefill_mode=env.get("CAREKV_PREFILL_MODE",
                              PAPER_BEST_ENV["CAREKV_PREFILL_MODE"]),
        residual_kind=env.get("CAREKV_PREFILL_RESIDUAL_KIND",
                               PAPER_BEST_ENV["CAREKV_PREFILL_RESIDUAL_KIND"]),
        store_budget_mode=env.get("CAREKV_STORE_BUDGET_MODE", "absolute"),
        read_budget_mode=env.get("CAREKV_READ_BUDGET_MODE", "absolute"),
        store_budget_ratio=env.get("CAREKV_STORE_BUDGET", "0"),
        read_budget_ratio=env.get("CAREKV_READ_BUDGET", "0"),
        store_abs_k=env.get("CAREKV_STORE_ABS_K", "0"),
        store_abs_v=env.get("CAREKV_STORE_ABS_V", "0"),
        read_abs_k=env.get("CAREKV_READ_ABS_K", "0"),
        read_abs_v=env.get("CAREKV_READ_ABS_V", "0"),
        ppl=round(ppl, 4),
        total_tokens=int(total_tokens),
        seconds=round(dt, 1),
        K_reads=int(stats.get("k_slots_read", 0)),
        V_reads=int(stats.get("v_slots_read", 0)),
        K_stored=int(stats.get("k_slots_stored", 0)),
        V_stored=int(stats.get("v_slots_stored", 0)),
        mean_dO_K=float(stats.get("mean_dO_K", 0.0) or 0.0),
        mean_dO_V=float(stats.get("mean_dO_V", 0.0) or 0.0),
        cache_mem_MB=round(mem_mb, 3),
        base_bits=base_bits,
        seq_len=seq_len,
        dataset="synthetic",
    )
    # Candidate-cap-aware effective-budget accounting (requested vs effective
    # SK/SV, caps, utilization, recovered elements, residual memory bytes).
    row.update(_effective_budget_fields(cc, hd, seq_len, stats))
    return row


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

EXPERIMENT_REGISTRY = {
    "A1": ("budget_ratio_grid", cells_A1_ratio),
    "A2": ("budget_absolute_grid", cells_A2_absolute),
    "B":  ("store_budget_sweep", cells_B_store),
    "C":  ("read_budget_sweep",  cells_C_read),
    "D":  ("kv_budget_balance",  cells_D_balance),
}


def _write_csv(rows, path):
    if not rows: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {len(rows)} rows -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    choices=list(EXPERIMENT_REGISTRY) + ["all"])
    ap.add_argument("--out-dir", required=True,
                    help="directory for output CSV(s)")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--base-bits", type=int, default=3)
    args = ap.parse_args()

    targets = (list(EXPERIMENT_REGISTRY) if args.experiment == "all"
               else [args.experiment])

    for key in targets:
        name, cell_fn = EXPERIMENT_REGISTRY[key]
        out_csv = os.path.join(args.out_dir, f"{name}.csv")
        print(f"\n========== experiment {key} → {name} ==========",
              flush=True)
        rows = []
        for label, env in cell_fn():
            try:
                r = run_cell(label, env, args.seq_len, args.base_bits)
                rows.append(r)
                print(f"[{key}] {label:36s} PPL={r['ppl']:.4f}  "
                      f"K_reads={r['K_reads']:>8d} V_reads={r['V_reads']:>8d}  "
                      f"mem={r['cache_mem_MB']:.2f} MB  ({r['seconds']:.1f}s)",
                      flush=True)
            except Exception as e:
                print(f"[{key}] {label} ERROR: {type(e).__name__}: {e}",
                      flush=True)
                rows.append(dict(label=label, error=str(e)))
        _write_csv(rows, out_csv)


if __name__ == "__main__":
    main()
