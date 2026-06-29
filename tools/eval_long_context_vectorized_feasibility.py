"""tools/eval_long_context_vectorized_feasibility.py — VPhase F.

Re-test long-context feasibility now that the vectorized correction gives
~110x speedup. The CARE-KV patch is Llama-only, so:
  • fp16 + BaseQuant INT3 run on real Qwen2.5-7B at SL=512 (per-channel/
    per-token KV fake-quant, as in the large-scale Part B).
  • vectorized CARE-KV (uniform INT3) runs on TinyLlama (supported Llama
    GQA) at SL=512 and SL=1024 — previously infeasible with the Python loop
    (~tens of GPU-hours/cell), now seconds-to-minutes.

Reports runtime + PPL + whether each setting is feasible.

Outputs:
  results/.../ablations/long_context_vectorized_carekv_feasibility.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time

import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from CARE_KV.care_kv.baselines import (BaseQuantAdapter, CAREKVAdapter,
                                       eval_ppl_wikitext)
from CARE_KV.care_kv.baselines.common import (DEVICE, measure_peak_gpu_mb,
                                              reset_peak_gpu)

TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _sym_quant(xr, bits, dim):
    qmax = float(2 ** (bits - 1) - 1)
    scale = (xr.abs().amax(dim=dim, keepdim=True) / qmax).clamp(min=1e-8)
    return (xr / scale).round().clamp(-qmax - 1, qmax) * scale


def qwen_kv_quant_hooks(model, bits, nkv, Dh):
    handles = []
    for name, mod in model.named_modules():
        is_k, is_v = name.endswith("k_proj"), name.endswith("v_proj")
        if not (is_k or is_v):
            continue
        def hook(m, inp, out, _b=bits, _k=is_k):
            B, T, HD = out.shape
            if HD != nkv * Dh:
                return out
            x = out.reshape(B, T, nkv, Dh).to(torch.float32)
            xq = _sym_quant(x, _b, dim=1) if _k else _sym_quant(x, _b, dim=-1)
            return xq.reshape(B, T, HD).to(out.dtype)
        handles.append(mod.register_forward_hook(hook))
    return handles


def run_qwen(model_id, bits, seq_len, n, tok, cfg):
    nah, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    Dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // nah
    reset_peak_gpu()
    t0 = time.perf_counter()
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                             device_map=DEVICE)
    m.eval()
    handles = qwen_kv_quant_hooks(m, bits, nkv, Dh) if bits < 16 else []
    ppl, ntok = eval_ppl_wikitext(m, tok, seq_len, n)
    for h in handles: h.remove()
    dt = time.perf_counter() - t0
    peak = measure_peak_gpu_mb()
    del m; torch.cuda.empty_cache()
    return ppl, dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--qwen", default="Qwen/Qwen2.5-7B")
    args = ap.parse_args()
    rows = []

    # ── Qwen2.5-7B SL=512: fp16 + BaseQuant INT3 ──
    qcfg = AutoConfig.from_pretrained(args.qwen)
    qtok = AutoTokenizer.from_pretrained(args.qwen)
    if qtok.pad_token_id is None: qtok.pad_token_id = qtok.eos_token_id or 0
    for label, bits in [("Qwen2.5-7B fp16", 16), ("Qwen2.5-7B BaseQuant_INT3", 3)]:
        try:
            ppl, dt, peak = run_qwen(args.qwen, bits, 512, 1, qtok, qcfg)
            rows.append(dict(model="Qwen2.5-7B", cell=label, seq_len=512, num_samples=1,
                             ppl=round(ppl, 4), runtime_s=round(dt, 1),
                             peak_gpu_MB=round(peak, 1), feasible="yes"))
            print(f"[F] {label:30s} SL512 PPL={ppl:.4f} rt={dt:.1f}s", flush=True)
        except Exception as e:
            rows.append(dict(model="Qwen2.5-7B", cell=label, seq_len=512, num_samples=1,
                             ppl="", runtime_s="", peak_gpu_MB="", feasible=f"ERROR:{type(e).__name__}"))
            print(f"[F] {label} ERROR: {e}", flush=True)

    # ── TinyLlama vectorized CARE-KV at SL=512 and SL=1024 (feasibility) ──
    ttok = AutoTokenizer.from_pretrained(TINY)
    if ttok.pad_token_id is None: ttok.pad_token_id = ttok.eos_token_id or 0
    # base anchors on TinyLlama for context
    for SL in (512, 1024):
        # BaseQuant INT3 (fast)
        bq = BaseQuantAdapter(bits=3)
        reset_peak_gpu(); t0 = time.perf_counter()
        m = bq.setup_model(TINY); ppl_bq, _ = eval_ppl_wikitext(m, ttok, SL, 1)
        dt_bq = time.perf_counter() - t0; del m; torch.cuda.empty_cache()
        rows.append(dict(model="TinyLlama", cell="BaseQuant_INT3", seq_len=SL, num_samples=1,
                         ppl=round(ppl_bq, 4), runtime_s=round(dt_bq, 1),
                         peak_gpu_MB="", feasible="yes"))
        print(f"[F] TinyLlama BaseQuant_INT3 SL{SL} PPL={ppl_bq:.4f} rt={dt_bq:.1f}s", flush=True)
        # vectorized CARE-KV (uniform INT3)
        mp = max(16, SL // 16 + 4)
        ck = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                           max_pages=mp, correction_impl="vectorized")
        reset_peak_gpu(); t0 = time.perf_counter()
        m = ck.setup_model(TINY); ppl_ck, _ = eval_ppl_wikitext(m, ttok, SL, 1)
        dt_ck = time.perf_counter() - t0
        st = ck.collect_debug_stats(); peak = measure_peak_gpu_mb()
        del m; torch.cuda.empty_cache()
        rows.append(dict(model="TinyLlama", cell="uniform_INT3_CAREKV_vectorized",
                         seq_len=SL, num_samples=1, ppl=round(ppl_ck, 4),
                         runtime_s=round(dt_ck, 1), peak_gpu_MB=round(peak, 1),
                         feasible="yes", k_reads=st["k_reads"], v_reads=st["v_reads"],
                         dppl_vs_basequant=round(ppl_ck - ppl_bq, 4)))
        print(f"[F] TinyLlama uniform_CAREKV_vec SL{SL} PPL={ppl_ck:.4f} "
              f"(Δvs base {ppl_ck-ppl_bq:+.4f}) rt={dt_ck:.1f}s K={st['k_reads']} V={st['v_reads']}",
              flush=True)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
