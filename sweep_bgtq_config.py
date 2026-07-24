"""Sweep CARE-KV residual configs on top of Block-GTQ base to maximize the PPL
improvement Δ=C−B. Calibrates Block-GTQ once, runs B (no residual) once, then
sweeps a list of C configs (granularity + budgets + K-correction mode).

Levers:
  - CAREKV_K_CORRECTION_MODE = linear | exact  (exact = renormalized softmax,
    documented free strict improvement)
  - k_channel_group / v_token_block  (smaller → more residual candidates →
    more correction capacity, but more residual memory)
  - sk/sv (store budget), rk/rv (read budget)

Usage:
  CUDA_VISIBLE_DEVICES=5 python sweep_bgtq_config.py \
      --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --seq-len 512 --num-samples 16
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

# (name, k_channel_group, v_token_block, sk, sv, rk, rv, kmode, residual_bits)
CONFIGS = [
    ("cur",         32, 4,  2,  4,  2,  2, "linear", 4),  # paper baseline
    ("cur_rb8",     32, 4,  2,  4,  2,  2, "linear", 8),  # 8-bit residual alone
    ("finer_rb4",    8, 1, 16, 16, 16, 16, "exact",  4),  # best config, 4-bit
    ("finer_rb8",    8, 1, 16, 16, 16, 16, "exact",  8),  # best config + 8-bit
    ("fine_rb8",    16, 2,  8,  8,  8,  8, "exact",  8),  # mid granularity + 8-bit
]


def run_cfg(model_id, sl, n, name, kcg, vtb, sk, sv, rk, rv, kmode, rb=4):
    os.environ["CAREKV_K_CORRECTION_MODE"] = kmode
    os.environ["CAREKV_RESIDUAL_BITS"] = str(rb)
    ad = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="blockgtq_style",
                       sk=sk, sv=sv, rk=rk, rv=rv,
                       k_channel_group=kcg, v_token_block=vtb,
                       k_store_mode="post_rope", correction_impl="vectorized")
    t0 = time.perf_counter()
    row = run_one(ad, model_id, "wikitext", sl, n)
    dt = time.perf_counter() - t0
    os.environ["CAREKV_K_CORRECTION_MODE"] = "linear"
    return row, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--append-csv", default="results/blockgtq_carekv/config_sweep.csv")
    a = ap.parse_args()

    print(f"[sweep] {a.model_id} SL={a.seq_len} N={a.num_samples}", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model_id)
    ct = "\n\n".join(r for r in load_dataset("wikitext","wikitext-2-raw-v1",split="train")["text"] if r.strip())
    calib = tok(ct, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :2048]
    m = LlamaForCausalLM.from_pretrained(a.model_id, torch_dtype=torch.float16, device_map="cuda").eval()
    bgb.reset()
    bgb.calibrate(m, calib, k_avg_bits=3.0, v_bits=3, device="cuda", n_calib_tokens=2048)
    import gc; m = m.to("meta"); del m; gc.collect(); torch.cuda.empty_cache()
    print("[sweep] calibrated", flush=True)

    # B (no residual) once
    Brow, _ = run_cfg(a.model_id, a.seq_len, a.num_samples, "B", 32, 4, 0, 0, 0, 0, "linear")
    B = Brow.ppl
    print(f"[sweep] B (Block-GTQ base) = {B}", flush=True)

    results = []
    for cfg in CONFIGS:
        name = cfg[0]
        row, dt = run_cfg(a.model_id, a.seq_len, a.num_samples, *cfg)
        C = row.ppl
        d = C - B
        results.append((name, C, d, 100*d/B, row.k_reads, row.v_reads, dt))
        print(f"[sweep] {name:14s} C={C:.4f} Δ={d:+.4f} ({100*d/B:+.3f}%) "
              f"reads K={row.k_reads}/V={row.v_reads} {dt:.0f}s", flush=True)

    os.makedirs(os.path.dirname(a.append_csv) or ".", exist_ok=True)
    wh = not os.path.exists(a.append_csv) or os.path.getsize(a.append_csv) == 0
    with open(a.append_csv, "a", newline="") as f:
        w = csv.writer(f)
        if wh: w.writerow(["model","seq_len","num_samples","config","B_ppl","C_ppl","delta","delta_pct","k_reads","v_reads","seconds"])
        for name, C, d, dp, kr, vr, dt in results:
            w.writerow([a.model_id, a.seq_len, a.num_samples, name, round(B,4), round(C,4), round(d,4), round(dp,3), kr, vr, round(dt,1)])
    print(f"\n[sweep] BEST: {min(results, key=lambda r: r[2])[0]}  "
          f"(Δ={min(r[2] for r in results):+.4f})", flush=True)


if __name__ == "__main__":
    main()
