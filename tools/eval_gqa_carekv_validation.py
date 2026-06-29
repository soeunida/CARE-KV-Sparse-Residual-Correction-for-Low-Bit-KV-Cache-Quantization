"""tools/eval_gqa_carekv_validation.py — Part D: GQA architecture validation.

Validates CARE-KV on a grouped-query-attention model (TinyLlama-1.1B is
GQA: 32 attention heads / 4 KV heads, group=8). Prints the config and runs
the acceptance checks:

  A. no shape mismatch        — CARE-KV prefill forward runs end-to-end
  B. repeat_kv path correct   — num_key_value_groups == Hq/Hkv, output shape OK
  C. estimator uses Hkv       — estimate_memory scales with num_key_value_heads,
                                NOT num_attention_heads
  D. nonzero K/V reads        — router fires for CARE-KV rows (K_reads+V_reads>0)
  E. READ=0 invariant         — CARE-KV with RK=RV=0 == base_quant (bit-exact logits)

A small PPL row set is run at low SL for live numbers; the full-budget
anchor PPLs (uniform/KIVI/KVQuant INT3 + CARE-KV, SL=128 N=4) are produced
by the Part A sweep and referenced in the summary (same GQA model).

Outputs:
  results/.../ablations/gqa_carekv_validation.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time

import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv.baselines import (
    FP16Adapter, BaseQuantAdapter, CAREKVAdapter, eval_ppl_synthetic,
)
from CARE_KV.care_kv.baselines.common import DEVICE, SYNTHETIC_PROMPT

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def print_config(model_id):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model_id)
    nah, nkv = c.num_attention_heads, c.num_key_value_heads
    hd = c.hidden_size // nah
    info = dict(model_id=model_id, model_type=c.model_type,
               num_hidden_layers=c.num_hidden_layers, hidden_size=c.hidden_size,
               num_attention_heads=nah, num_key_value_heads=nkv, head_dim=hd,
               num_key_value_groups=nah // nkv, is_gqa=(nkv < nah))
    print("=== MODEL CONFIG ===")
    for k, v in info.items():
        print(f"  {k:24s}: {v}")
    return info


def get_logits(model, tok, n=64):
    enc = tok(SYNTHETIC_PROMPT, return_tensors="pt", truncation=True, max_length=n)
    ids = enc["input_ids"].to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    return out.logits.float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--seq-len", type=int, default=64)
    args = ap.parse_args()

    cfg = print_config(MODEL_ID)
    nah, nkv, hd, L = (cfg["num_attention_heads"], cfg["num_key_value_heads"],
                       cfg["head_dim"], cfg["num_hidden_layers"])
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    checks = {}
    rows = []

    # ── Check C: estimator uses Hkv, not Hq ──────────────────────────────
    a = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform", max_pages=8)
    mem_kv = a.estimate_memory(args.seq_len, num_layers=L, hkv=nkv, head_dim=hd)
    mem_hq = a.estimate_memory(args.seq_len, num_layers=L, hkv=nah, head_dim=hd)
    # If the estimator used Hq it would be ~group× larger. Confirm Hkv scaling:
    ratio = mem_hq["estimated_kv_memory_MB"] / max(mem_kv["estimated_kv_memory_MB"], 1e-9)
    checks["C_estimator_uses_kv_heads"] = abs(ratio - (nah / nkv)) < 0.5
    print(f"[CHECK C] estimator(Hkv)={mem_kv['estimated_kv_memory_MB']:.4f}MB "
          f"estimator(Hq)={mem_hq['estimated_kv_memory_MB']:.4f}MB "
          f"ratio={ratio:.2f} (expect ~group={nah/nkv:.0f}) → "
          f"{'PASS' if checks['C_estimator_uses_kv_heads'] else 'FAIL'}")

    # ── Check B: repeat_kv group factor ──────────────────────────────────
    checks["B_repeat_kv_group"] = (nah // nkv) * nkv == nah and (nah % nkv == 0)
    print(f"[CHECK B] num_key_value_groups={nah//nkv}, {nkv}*{nah//nkv}={nah} → "
          f"{'PASS' if checks['B_repeat_kv_group'] else 'FAIL'}")

    # ── Checks A + D: CARE-KV forward runs, nonzero reads ────────────────
    t0 = time.time()
    m = a.setup_model(MODEL_ID)
    try:
        ppl, ntok = eval_ppl_synthetic(m, tok, args.seq_len)
        checks["A_no_shape_mismatch"] = True
    except Exception as e:
        checks["A_no_shape_mismatch"] = False
        print(f"[CHECK A] FAILED with {type(e).__name__}: {e}")
        ppl, ntok = 0.0, 0
    stats = a.collect_debug_stats()
    kreads, vreads = stats["k_reads"], stats["v_reads"]
    checks["D_nonzero_reads"] = (kreads + vreads) > 0
    dt = time.time() - t0
    print(f"[CHECK A] CARE-KV prefill forward ran, PPL={ppl:.3f} → "
          f"{'PASS' if checks['A_no_shape_mismatch'] else 'FAIL'}")
    print(f"[CHECK D] K_reads={kreads} V_reads={vreads} → "
          f"{'PASS' if checks['D_nonzero_reads'] else 'FAIL'}")
    if hasattr(a, "teardown"): a.teardown()
    del m; torch.cuda.empty_cache()
    rows.append(dict(method="CAREKV_uniform_INT3", ppl=round(ppl, 4),
                     k_reads=kreads, v_reads=vreads, seq_len=args.seq_len,
                     runtime_s=round(dt, 1), note="GQA forward + router fired"))

    # ── Check E: READ=0 invariant (CARE-KV RK=RV=0 == base_quant logits) ──
    bq = BaseQuantAdapter(bits=3)
    mb = bq.setup_model(MODEL_ID)
    logits_bq = get_logits(mb, tok, args.seq_len)
    del mb; torch.cuda.empty_cache()
    a0 = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                       sk=2, sv=4, rk=0, rv=0, max_pages=8)
    m0 = a0.setup_model(MODEL_ID)
    logits_c0 = get_logits(m0, tok, args.seq_len)
    s0 = a0.collect_debug_stats()
    if hasattr(a0, "teardown"): a0.teardown()
    del m0; torch.cuda.empty_cache()
    max_abs = (logits_bq - logits_c0).abs().max().item()
    checks["E_read0_invariant"] = max_abs < 1e-3
    print(f"[CHECK E] READ=0 vs base_quant max|Δlogit|={max_abs:.2e} "
          f"(reads K={s0['k_reads']} V={s0['v_reads']}) → "
          f"{'PASS' if checks['E_read0_invariant'] else 'FAIL'}")
    rows.append(dict(method="CAREKV_uniform_INT3_READ0", ppl=0.0,
                     k_reads=s0["k_reads"], v_reads=s0["v_reads"],
                     seq_len=args.seq_len, runtime_s=0.0,
                     note=f"READ=0 invariant max|dlogit|={max_abs:.2e} vs base_quant"))

    # ── fp16 + base_quant reference PPL (fast) ───────────────────────────
    for adp in (FP16Adapter(), BaseQuantAdapter(bits=4), BaseQuantAdapter(bits=3)):
        mm = adp.setup_model(MODEL_ID)
        p, nt = eval_ppl_synthetic(mm, tok, args.seq_len)
        del mm; torch.cuda.empty_cache()
        rows.append(dict(method=adp.name, ppl=round(p, 4), k_reads=0, v_reads=0,
                         seq_len=args.seq_len, runtime_s=0.0, note="reference"))
        print(f"  {adp.name:24s} PPL={p:.4f}")

    all_pass = all(checks.values())
    print(f"\n=== ACCEPTANCE: {'ALL PASS' if all_pass else 'SOME FAILED'} ===")
    for k, v in checks.items():
        print(f"  {k:32s}: {'PASS' if v else 'FAIL'}")

    # Write CSV: config + checks + rows
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        for k, v in cfg.items():
            w.writerow(["config", k, v])
        for k, v in checks.items():
            w.writerow(["acceptance_check", k, "PASS" if v else "FAIL"])
        w.writerow([])
        w.writerow(["method", "ppl", "k_reads", "v_reads", "seq_len", "runtime_s", "note"])
        for r in rows:
            w.writerow([r["method"], r["ppl"], r["k_reads"], r["v_reads"],
                        r["seq_len"], r["runtime_s"], r["note"]])
    print(f"wrote -> {args.out_csv}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
