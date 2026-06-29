"""tools/profile_carekv_decode_overhead.py — Part E.

Quantify the floating-point + wall-clock overhead of CARE-KV's residual
correction during decode, to answer:
  1. How much extra FLOP does CARE-KV add?
  2. Is overhead dominated by K correction, V correction, or router scoring?
  3. Is the current runtime dominated by Python loops?
  4. What needs to be fused/vectorized for practical decode?
  5. Does overhead grow with seq_len, batch, or residual budget?

Two kinds of numbers, clearly separated:
  • ANALYTICAL FLOP model (correction vs base attention), parameterized by
    config × seq_len × decode_tokens × budget. Exact arithmetic counts.
  • MEASURED wall-clock (prefill + decode) for base_quant vs CARE-KV at a
    tractable scale, showing the Python-loop gap. CARE-KV's prototype decode
    re-prefills per token (Phase G v1), so measured decode is run at small
    new_tokens and the per-token cost is reported honestly.

Outputs:
  results/.../ablations/carekv_decode_overhead.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time

import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer
from CARE_KV.care_kv.baselines import BaseQuantAdapter, CAREKVAdapter, FP16Adapter
from CARE_KV.care_kv.baselines.common import DEVICE, SYNTHETIC_PROMPT

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def flop_model(L, Hq, Dh, T, sk, sv, rk, rv, sketch_dim=16):
    """Per-decode-token FLOP (MAC) counts, summed over layers.

    Base attention (per token): QK^T + AV ≈ 2·Hq·T·Dh MACs (+ softmax Hq·T).
    CARE-KV correction (per token):
      K-correction: for each query head, RK residual slots, each a q·r_k dot
        product over Dh  → rk·Hq·Dh MACs (+ add to logits).
      V-correction: RV residual slots, weighted residual addition over Dh
        → rv·Hq·Dh MACs.
      Router scoring: score the candidate stored slots feeding the read.
        Candidate pool ≈ (sk+sv) stored slots/token over the context; each
        scored with a sketch_dim inner product → (sk+sv)·sketch_dim·Hq MACs
        (per-token, dominated by current context — upper-bounded here).
    """
    base = L * (2 * Hq * T * Dh)
    softmax = L * (Hq * T)
    k_corr = L * (rk * Hq * Dh)
    v_corr = L * (rv * Hq * Dh)
    router = L * ((sk + sv) * sketch_dim * Hq)
    corr_total = k_corr + v_corr + router
    return dict(base_attn_MAC=base, softmax_FLOP=softmax,
                k_corr_MAC=k_corr, v_corr_MAC=v_corr, router_MAC=router,
                corr_total_MAC=corr_total,
                corr_over_base_pct=100.0 * corr_total / max(base, 1),
                k_share_pct=100.0 * k_corr / max(corr_total, 1),
                v_share_pct=100.0 * v_corr / max(corr_total, 1),
                router_share_pct=100.0 * router / max(corr_total, 1))


def counters(L, Hq, Dh, decode_tokens, rk, rv):
    """Lightweight per-run correction counters."""
    qrk = L * Hq * rk * decode_tokens                       # q·R_K dot products
    vadd = L * Hq * rv * decode_tokens                      # V residual weighted additions
    k_elems = qrk * Dh                                      # K residual elements read
    v_elems = vadd * Dh                                     # V residual elements read
    # residual stored at ~4-bit packed → 0.5 byte/elem + small index
    resid_bytes = (k_elems + v_elems) * 0.5
    softmax_jac = L * Hq * decode_tokens                    # softmax/Jacobian correction ops
    return dict(qRk_dot_products=qrk, V_resid_weighted_adds=vadd,
                softmax_jacobian_ops=softmax_jac,
                K_resid_elems_read=k_elems, V_resid_elems_read=v_elems,
                total_resid_bytes_read=int(resid_bytes))


def measure(adapter, tok, prompt_len, new_tokens):
    """Measure prefill + decode wall-clock via use_cache=True."""
    m = adapter.setup_model(MODEL_ID)
    # streaming decode needs use_cache; CARE-KV supports it (Phase G)
    m.config.use_cache = True
    enc = tok(SYNTHETIC_PROMPT, return_tensors="pt", truncation=True, max_length=prompt_len)
    ids = enc["input_ids"].to(DEVICE)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    if DEVICE == "cuda": torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    if DEVICE == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(new_tokens):
            o = m(input_ids=nxt, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = o.logits[:, -1:].argmax(-1)
    if DEVICE == "cuda": torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0
    if hasattr(adapter, "teardown"): adapter.teardown()
    del m
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return dict(prefill_ms=round(prefill_s * 1000, 1),
                decode_ms_per_token=round(decode_s / max(new_tokens, 1) * 1000, 1),
                tokens_per_sec=round(new_tokens / max(decode_s, 1e-9), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--flop-seqlen", type=int, default=1024)
    ap.add_argument("--flop-decode-tokens", type=int, default=64)
    ap.add_argument("--measure-prompt", type=int, default=256)
    ap.add_argument("--measure-decode", type=int, default=4)
    ap.add_argument("--skip-measure", action="store_true")
    args = ap.parse_args()

    # TinyLlama GQA config
    L, Hq, Dh = 22, 32, 64
    rows = []

    # ── Analytical FLOP model at the requested scale, across budgets ──
    print(f"=== ANALYTICAL FLOP (per decode token, SL={args.flop_seqlen}, "
          f"TinyLlama L={L} Hq={Hq} Dh={Dh}) ===")
    for (sk, sv, rk, rv) in [(1, 2, 1, 1), (2, 4, 2, 2), (4, 4, 2, 2)]:
        fm = flop_model(L, Hq, Dh, args.flop_seqlen, sk, sv, rk, rv)
        ct = counters(L, Hq, Dh, args.flop_decode_tokens, rk, rv)
        print(f"  SK{sk}SV{sv}RK{rk}RV{rv}: corr/base={fm['corr_over_base_pct']:.3f}%  "
              f"(K {fm['k_share_pct']:.0f}% / V {fm['v_share_pct']:.0f}% / "
              f"router {fm['router_share_pct']:.0f}%)")
        rows.append(dict(kind="analytical_flop", config=f"SK{sk}SV{sv}RK{rk}RV{rv}",
                         seq_len=args.flop_seqlen, decode_tokens=args.flop_decode_tokens,
                         **fm, **ct,
                         prefill_ms="", decode_ms_per_token="", tokens_per_sec=""))

    # ── Scaling check: corr/base vs seq_len (budget fixed RK=RV=2) ──
    for T in (128, 512, 1024, 2048):
        fm = flop_model(L, Hq, Dh, T, 2, 4, 2, 2)
        rows.append(dict(kind="flop_vs_seqlen", config="SK2SV4RK2RV2",
                         seq_len=T, decode_tokens=args.flop_decode_tokens, **fm,
                         **counters(L, Hq, Dh, args.flop_decode_tokens, 2, 2),
                         prefill_ms="", decode_ms_per_token="", tokens_per_sec=""))

    # ── Measured wall-clock: base_quant vs CARE-KV (small scale) ──
    if not args.skip_measure:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        print(f"\n=== MEASURED wall-clock (prompt={args.measure_prompt}, "
              f"decode={args.measure_decode}) ===")
        specs = [
            ("fp16", FP16Adapter()),
            ("base_quant_INT3", BaseQuantAdapter(bits=3)),
            ("CAREKV_uniform_INT3_SK2SV4", CAREKVAdapter(mode="fixed", bits=3,
                base_quantizer="uniform", max_pages=64)),
        ]
        for name, adp in specs:
            try:
                mres = measure(adp, tok, args.measure_prompt, args.measure_decode)
            except Exception as e:
                import traceback; traceback.print_exc()
                mres = dict(prefill_ms=0, decode_ms_per_token=0, tokens_per_sec=0)
            print(f"  {name:30s} prefill={mres['prefill_ms']:.0f}ms "
                  f"decode/token={mres['decode_ms_per_token']:.0f}ms "
                  f"tok/s={mres['tokens_per_sec']:.3f}")
            rows.append(dict(kind="measured_walltime", config=name,
                             seq_len=args.measure_prompt,
                             decode_tokens=args.measure_decode,
                             base_attn_MAC="", softmax_FLOP="", k_corr_MAC="",
                             v_corr_MAC="", router_MAC="", corr_total_MAC="",
                             corr_over_base_pct="", k_share_pct="", v_share_pct="",
                             router_share_pct="", qRk_dot_products="",
                             V_resid_weighted_adds="", softmax_jacobian_ops="",
                             K_resid_elems_read="", V_resid_elems_read="",
                             total_resid_bytes_read="", **mres))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
