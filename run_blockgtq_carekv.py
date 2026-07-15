"""run_blockgtq_carekv.py — Block-GTQ base ⊕ CARE-KV residual, WikiText-2 PPL.

Measures, under identical conditions (model / seq_len / calib / seed):

  B = Block-GTQ K3V3 baseline PPL          (base quantization only)
  C = Block-GTQ K3V3 + CARE-KV residual    (CARE-KV store+router on top)
  Δ = C - B

Three modes (all share the SAME Block-GTQ per-(layer,head) calibration):

  standalone   : raw HF model + Block-GTQ unaccelerated patch (independent
                 baseline, does not touch CARE-KV) — cross-check for B.
  base_quant   : CARE-KV patch, base_quantizer=blockgtq_style, residual OFF
                 → B via CARE-KV (should match `standalone`).
  carekv       : CARE-KV patch, base_quantizer=blockgtq_style, residual ON
                 (paper-locked STORE/READ budgets) → C.

No fp16 recent-key buffer in any mode (unaccelerated path / carekv_stored
full-prefill; no PM-KVQ side window).

Env / CLI:
  MODEL_ID, SEQ_LEN, NUM_SAMPLES, K_AVG_BITS(=3), V_BITS(=3),
  CALIB_TOKENS(=2048), SEED(=0)
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time

# Limit CPU threads BEFORE importing torch to avoid oversubscription when many
# of these (CPU-bound residual-path) jobs run in parallel across GPUs.
_NT = os.environ.get("CAREKV_NUM_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _NT)

import torch
torch.set_num_threads(int(_NT))

sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/blockgtq")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _wt2_text(split):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return [r for r in ds["text"] if r.strip()]


def _eval_ppl(model, chunks):
    """Windowed non-overlapping PPL. Resets CARE-KV per-window."""
    from CARE_KV.care_kv import get_debug_stats, reset_debug_stats
    reset_debug_stats()
    if DEVICE == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    N, T = chunks.shape
    total_loss, total_tok = 0.0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N):
            _tw = time.perf_counter()
            ids = chunks[i:i+1].to(DEVICE)
            out = model(input_ids=ids, labels=ids, use_cache=False)
            n = T - 1
            total_loss += float(out.loss.item()) * n
            total_tok += n
            if os.environ.get("CAREKV_WINDOW_TIMING", "0") == "1":
                print(f"[win {i+1}/{N}] {time.perf_counter()-_tw:.1f}s", flush=True)
            for sub in model.modules():
                if hasattr(sub, "reset_cache") and hasattr(sub, "_caches"):
                    sub.reset_cache()
    if DEVICE == "cuda": torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ppl = math.exp(total_loss / total_tok)
    st = get_debug_stats()
    peak = (torch.cuda.max_memory_allocated()/1e6) if DEVICE == "cuda" else 0.0
    return dict(ppl=round(ppl, 6), total_tokens=total_tok, seconds=round(dt, 2),
                K_reads=st.get("k_slots_read", 0), V_reads=st.get("v_slots_read", 0),
                peak_gpu_mem_MB=round(peak, 2))


def _build_carekv(model, base_bits, residual_on):
    from CARE_KV.care_kv import (CacheConfig, patch_llama_model,
                                 reset_all_caches, apply_carekv_env_overrides)
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    kw = dict(
        num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=base_bits,
        group_size=32, k_channel_group=32, page_size=16, max_pages=512,
        v_token_block=4, sketch_dim=32,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
        packed_base=True, scale_quant="int8",
        route_policy="joint", correction_impl="cached", budget_policy="uniform",
        base_quantizer="blockgtq_style",
    )
    if not residual_on:
        # B: no residual — force all budgets to 0 so output == Block-GTQ base.
        kw.update(store_abs_k=0, store_abs_v=0, read_abs_k=0, read_abs_v=0)
        os.environ["CAREKV_PREFILL_MODE"] = "base_quant"
    else:
        os.environ["CAREKV_PREFILL_MODE"] = "carekv_stored"
        os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
    os.environ["CAREKV_BASE_QUANTIZER"] = "blockgtq_style"
    apply_carekv_env_overrides(kw)
    cc = CacheConfig(**kw)
    model = patch_llama_model(model, cc); reset_all_caches(model); model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=os.environ.get("MODEL_ID",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    ap.add_argument("--mode", required=True,
                    choices=["standalone", "base_quant", "carekv"])
    ap.add_argument("--seq-len", type=int, default=int(os.environ.get("SEQ_LEN","512")))
    ap.add_argument("--num-samples", type=int, default=int(os.environ.get("NUM_SAMPLES","32")))
    ap.add_argument("--k-avg-bits", type=float, default=float(os.environ.get("K_AVG_BITS","3")))
    ap.add_argument("--v-bits", type=int, default=int(os.environ.get("V_BITS","3")))
    ap.add_argument("--calib-tokens", type=int, default=int(os.environ.get("CALIB_TOKENS","2048")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED","0")))
    ap.add_argument("--append-csv", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"[bgtq-carekv] model={args.model_id} mode={args.mode} "
          f"SL={args.seq_len} N={args.num_samples} K{int(args.k_avg_bits)}V{args.v_bits} "
          f"seed={args.seed}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None or tok.pad_token_id < 0:
        tok.pad_token_id = tok.eos_token_id or 0

    # Eval windows: WT2 test, non-overlapping.
    full = "\n\n".join(_wt2_text("test"))
    ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    need = args.seq_len * args.num_samples
    assert ids.numel() >= need, f"WT2 too short: {ids.numel()} < {need}"
    chunks = ids[:need].view(args.num_samples, args.seq_len)
    print(f"[bgtq-carekv] eval windows: {chunks.shape[0]} x {chunks.shape[1]}", flush=True)

    # Calib ids: WT2 TRAIN (disjoint from eval), first calib_tokens.
    ctext = "\n\n".join(_wt2_text("train"))
    calib_ids = tok(ctext, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :args.calib_tokens]

    model = LlamaForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None).eval()
    model.config.use_cache = False

    # Calibrate Block-GTQ once (all modes share it).
    import CARE_KV.care_kv.blockgtq_base as bgb
    bgb.reset()
    bgb.calibrate(model, calib_ids, k_avg_bits=args.k_avg_bits, v_bits=args.v_bits,
                  device=DEVICE, n_calib_tokens=args.calib_tokens)
    print(f"[bgtq-carekv] Block-GTQ calibrated: {bgb._REG['meta']}", flush=True)

    if args.mode == "standalone":
        from blockgtq.unaccelerated import (build_unaccelerated_quantizers,
                                            patch_model_kv, unpatch_model_kv)
        # Rebuild quantizers from same registry meta (reuse calibrated ones).
        kq, vq = bgb._REG["kq"], bgb._REG["vq"]
        hd = model.config.hidden_size // model.config.num_attention_heads
        nkv = model.config.num_key_value_heads
        handles = patch_model_kv(model, kq, vq, hd, nkv)
        res = _eval_ppl(model, chunks)
        unpatch_model_kv(handles)
        label = "standalone_blockgtq"
    else:
        residual_on = (args.mode == "carekv")
        model = _build_carekv(model, base_bits=int(args.k_avg_bits),
                              residual_on=residual_on)
        res = _eval_ppl(model, chunks)
        label = "carekv_blockgtq" if residual_on else "base_quant_blockgtq"

    row = dict(model=args.model_id, mode=label, seq_len=args.seq_len,
               num_samples=args.num_samples, k_avg_bits=args.k_avg_bits,
               v_bits=args.v_bits, calib_tokens=args.calib_tokens, seed=args.seed,
               **res)
    write_header = not os.path.exists(args.append_csv) or os.path.getsize(args.append_csv) == 0
    os.makedirs(os.path.dirname(args.append_csv) or ".", exist_ok=True)
    with open(args.append_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"[bgtq-carekv] {label}  PPL={res['ppl']:.4f}  tok={res['total_tokens']}  "
          f"{res['seconds']:.1f}s  K_reads={res['K_reads']} V_reads={res['V_reads']}  "
          f"peak={res['peak_gpu_mem_MB']}MB", flush=True)


if __name__ == "__main__":
    main()
