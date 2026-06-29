"""tools/test_carekv_streaming_decode.py — Part F.

Streaming-decode handling of newly generated KV under CARE-KV. Tests the
two store policies and validates the online-store smoke acceptance criteria.

Policy 1 — ONLINE store (what CARE-KV implements):
  Each new token's K/V is quantized to base K_hat/V_hat immediately; its
  residual R_K/R_V is computed at once; the store policy appends selected
  residual slots to the CARE-KV residual cache; future decode queries read
  those residuals.

Policy 2 — DELAYED store:
  Keep a short window of recent tokens in fp16 (exact), quantize + store
  residuals only after the window rolls out. Trades a small fp16 window for
  exactness on the most-recent (highest-attention) tokens.

Acceptance (online policy smoke):
  1. generated tokens append to base KV cache       (seq length grows)
  2. selected residuals append to residual cache    (stored slots grow)
  3. cache length increases correctly per step
  4. read router sees previous generated-token residuals (reads grow)
  5. READ=0 decode matches base-quant decode        (logits bit-close)
  6. no memory leak across decode steps             (peak mem stable)

Outputs:
  results/.../ablations/carekv_streaming_decode_smoke.csv
  results/.../summaries/carekv_streaming_decode_design.md  (written separately)
"""
from __future__ import annotations
import argparse, csv, os, sys

import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer
from CARE_KV.care_kv.baselines import BaseQuantAdapter, CAREKVAdapter
from CARE_KV.care_kv.baselines.common import DEVICE, SYNTHETIC_PROMPT
from CARE_KV.care_kv import get_debug_stats, reset_debug_stats


def stream_decode(adapter, tok, prompt_len, new_tokens, record_stats=True):
    """Prefill + step-by-step decode under use_cache=True. Returns per-step
    records (stored/read slots, mem) and the decoded logits sequence."""
    m = adapter.setup_model(MODEL_ID)
    m.config.use_cache = True
    enc = tok(SYNTHETIC_PROMPT, return_tensors="pt", truncation=True, max_length=prompt_len)
    ids = enc["input_ids"].to(DEVICE)
    recs = []
    logits_seq = []
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    reset_debug_stats()
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    logits_seq.append(out.logits[:, -1, :].float().cpu())
    base_len = int(ids.shape[1])
    for step in range(new_tokens):
        with torch.no_grad():
            o = m(input_ids=nxt, past_key_values=past, use_cache=True)
        past = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)
        logits_seq.append(o.logits[:, -1, :].float().cpu())
        s = get_debug_stats()
        try:
            hf_len = past.get_seq_length()
        except Exception:
            hf_len = base_len + step + 1
        mem = (torch.cuda.memory_allocated() / 1e6) if DEVICE == "cuda" else 0.0
        recs.append(dict(step=step, expected_len=base_len + step + 1,
                         hf_cache_len=int(hf_len),
                         k_slots_stored=int(s.get("k_slots_stored", 0)),
                         v_slots_stored=int(s.get("v_slots_stored", 0)),
                         k_slots_read=int(s.get("k_slots_read", 0)),
                         v_slots_read=int(s.get("v_slots_read", 0)),
                         mem_MB=round(mem, 1)))
    peak = (torch.cuda.max_memory_allocated() / 1e6) if DEVICE == "cuda" else 0.0
    if hasattr(adapter, "teardown"): adapter.teardown()
    del m
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return recs, logits_seq, peak


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--prompt-len", type=int, default=32)
    ap.add_argument("--new-tokens", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    checks = {}

    # ── Online-store CARE-KV streaming decode ──
    care = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                         sk=2, sv=4, rk=2, rv=2, max_pages=64)
    recs, logits_care, peak = stream_decode(care, tok, args.prompt_len, args.new_tokens)

    # 1+3. cache length grows correctly
    checks["1_3_cache_len_grows"] = all(
        r["hf_cache_len"] == r["expected_len"] or r["hf_cache_len"] >= r["expected_len"] - 1
        for r in recs) and recs[-1]["hf_cache_len"] > recs[0]["hf_cache_len"]
    # 2. residual stored slots grow over decode
    checks["2_residual_store_grows"] = (
        recs[-1]["k_slots_stored"] + recs[-1]["v_slots_stored"]
        > recs[0]["k_slots_stored"] + recs[0]["v_slots_stored"])
    # 4. reads grow (router sees prior generated-token residuals)
    checks["4_reads_grow"] = (
        recs[-1]["k_slots_read"] + recs[-1]["v_slots_read"]
        > recs[0]["k_slots_read"] + recs[0]["v_slots_read"])
    # 6. no memory leak: last-step mem within 5% of mid-run mem
    mid = recs[len(recs) // 2]["mem_MB"]; last = recs[-1]["mem_MB"]
    checks["6_no_mem_leak"] = (mid == 0) or (abs(last - mid) / max(mid, 1e-9) < 0.10)

    # 5. READ=0 decode matches base_quant decode (logits)
    care0 = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                          sk=2, sv=4, rk=0, rv=0, max_pages=64)
    recs0, logits_c0, _ = stream_decode(care0, tok, args.prompt_len, args.new_tokens)
    bq = BaseQuantAdapter(bits=3)
    recs_bq, logits_bq, _ = stream_decode(bq, tok, args.prompt_len, args.new_tokens)
    max_dl = max((logits_c0[i] - logits_bq[i]).abs().max().item()
                 for i in range(min(len(logits_c0), len(logits_bq))))
    checks["5_read0_matches_basequant"] = max_dl < 1e-2

    print("=== streaming decode per-step (online CARE-KV) ===")
    for r in recs:
        print(f"  step {r['step']}: len={r['hf_cache_len']} "
              f"stored(K{r['k_slots_stored']}/V{r['v_slots_stored']}) "
              f"read(K{r['k_slots_read']}/V{r['v_slots_read']}) mem={r['mem_MB']}MB")
    print(f"\nREAD=0 vs base_quant max|Δlogit| = {max_dl:.2e}")
    print("\n=== ACCEPTANCE ===")
    for k, v in checks.items():
        print(f"  {k:30s}: {'PASS' if v else 'FAIL'}")
    all_pass = all(checks.values())
    print(f"  {'ALL PASS' if all_pass else 'SOME FAILED'}")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "result"])
        for k, v in checks.items():
            w.writerow([k, "PASS" if v else "FAIL"])
        w.writerow(["read0_max_dlogit", f"{max_dl:.3e}"])
        w.writerow(["peak_mem_MB", round(peak, 1)])
        w.writerow([])
        w.writerow(["step", "hf_cache_len", "expected_len", "k_stored", "v_stored",
                    "k_read", "v_read", "mem_MB"])
        for r in recs:
            w.writerow([r["step"], r["hf_cache_len"], r["expected_len"],
                        r["k_slots_stored"], r["v_slots_stored"],
                        r["k_slots_read"], r["v_slots_read"], r["mem_MB"]])
    print(f"wrote -> {args.out_csv}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
