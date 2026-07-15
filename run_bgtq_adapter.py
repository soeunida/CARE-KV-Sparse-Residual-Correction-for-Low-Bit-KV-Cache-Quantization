"""Block-GTQ ⊕ CARE-KV via the FAST paper harness (run_one + CAREKVAdapter).

The hand-rolled full-prefill driver was ~80x slower than the paper's adapter
path for identical config/reads; this uses the adapter path the paper uses for
its SL512 multi-model runs (correction_impl=vectorized).

Per (model, seq_len): calibrate Block-GTQ once (post-RoPE), then run two arms
through run_one with base_quantizer=blockgtq_style:
  B = budgets 0/0/0/0   → Block-GTQ K3V3 baseline (no residual)
  C = SK2 SV4 RK2 RV2   → + CARE-KV residual (paper-locked budgets)
Δ = C - B. Both arms share the same harness, windows, tokenization → fair Δ.

Usage:
  CUDA_VISIBLE_DEVICES=5 python run_bgtq_adapter.py \
      --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 512 \
      --num-samples 8 --append-csv results/blockgtq_carekv/adapter.csv
"""
import argparse, csv, os, sys, time
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/CARE_KV/care_kv/tools")
sys.path.insert(0, "/home/soeun/blockgtq")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from CARE_KV.care_kv.baselines import CAREKVAdapter
from eval_base_quantizer_expansion import run_one
import CARE_KV.care_kv.blockgtq_base as bgb


def calibrate(model_id, device="cuda", k_avg_bits=3.0, v_bits=3, n_calib=2048):
    tok = AutoTokenizer.from_pretrained(model_id)
    ct = "\n\n".join(r for r in load_dataset("wikitext","wikitext-2-raw-v1",split="train")["text"] if r.strip())
    calib = tok(ct, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :n_calib]
    # Large models (>20B) don't fit one GPU — shard for calibration too.
    dmap = "auto" if os.environ.get("CAREKV_DEVICE_MAP") == "auto" else device
    m = LlamaForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=dmap,
        low_cpu_mem_usage=True).eval()
    # collect_qk_activations moves ids to `device`; for a sharded model the
    # embedding lives on cuda:0, so cuda:0 (="cuda") is the right entry device.
    bgb.reset()
    bgb.calibrate(m, calib, k_avg_bits=k_avg_bits, v_bits=v_bits, device="cuda", n_calib_tokens=n_calib)
    # Free the calibration model FULLY before run_one loads a second copy — for
    # large sharded models the two 68GB copies otherwise coexist and force
    # accelerate to offload layers to meta (→ empty k_proj weights → NaN) or OOM.
    import gc
    m = m.to("meta")
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return bgb._REG["meta"]


def one_arm(model_id, sl, n, label, sk, sv, rk, rv):
    ad = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="blockgtq_style",
                       sk=sk, sv=sv, rk=rk, rv=rv,
                       k_store_mode="post_rope", correction_impl="vectorized")
    t0 = time.perf_counter()
    row = run_one(ad, model_id, "wikitext", sl, n)
    dt = time.perf_counter() - t0
    print(f"[{label}] ppl={row.ppl}  k_reads={row.k_reads} v_reads={row.v_reads}  "
          f"{dt:.1f}s  notes={row.notes[:60]}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--num-samples", type=int, required=True)
    ap.add_argument("--append-csv", required=True)
    a = ap.parse_args()

    print(f"[bgtq-adapter] {a.model_id} SL={a.seq_len} N={a.num_samples}", flush=True)
    tc = time.perf_counter()
    meta = calibrate(a.model_id)
    print(f"[bgtq-adapter] calibrated {meta} in {time.perf_counter()-tc:.1f}s", flush=True)

    import gc
    B = one_arm(a.model_id, a.seq_len, a.num_samples, "B base_quant", 0, 0, 0, 0)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    C = one_arm(a.model_id, a.seq_len, a.num_samples, "C carekv",     2, 4, 2, 2)
    delta = (C.ppl - B.ppl) if (B.ppl and C.ppl) else None

    row = dict(model=a.model_id, seq_len=a.seq_len, num_samples=a.num_samples,
               B_ppl=B.ppl, C_ppl=C.ppl, delta=round(delta, 4) if delta is not None else "",
               delta_pct=round(100*delta/B.ppl, 3) if delta is not None and B.ppl else "",
               C_k_reads=C.k_reads, C_v_reads=C.v_reads,
               B_k_reads=B.k_reads, B_v_reads=B.v_reads,
               B_runtime_s=B.runtime_seconds, C_runtime_s=C.runtime_seconds)
    os.makedirs(os.path.dirname(a.append_csv) or ".", exist_ok=True)
    wh = not os.path.exists(a.append_csv) or os.path.getsize(a.append_csv) == 0
    with open(a.append_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if wh: w.writeheader()
        w.writerow(row)
    print(f"[bgtq-adapter] DONE  B={B.ppl}  C={C.ppl}  Δ={row['delta']} ({row['delta_pct']}%)  "
          f"C_reads K={C.k_reads}/V={C.v_reads}", flush=True)


if __name__ == "__main__":
    main()
