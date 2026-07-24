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
    # C-arm residual config is env-tunable (defaults = paper). Set
    # CAREKV_CFG_KCG/VTB/KMODE/RB to enable the improved configs (e.g. fine_rb8:
    # KCG=16 VTB=2 KMODE=exact RB=8 with sk/sv/rk/rv=8). B-arm (budgets 0) is
    # unaffected since it reads no residual.
    kcg = int(os.environ.get("CAREKV_CFG_KCG", "32"))
    vtb = int(os.environ.get("CAREKV_CFG_VTB", "4"))
    os.environ["CAREKV_K_CORRECTION_MODE"] = os.environ.get("CAREKV_CFG_KMODE", "linear")
    os.environ["CAREKV_RESIDUAL_BITS"] = os.environ.get("CAREKV_CFG_RB", "4")
    # Correction kernel is env-tunable (default vectorized = paper). The
    # vectorized path CPU-spins/hangs on Llama-2-13b's C-arm (40-layer full-MHA)
    # regardless of GPU/sharding; `cached` is CARE_KV's stable path and is
    # numerically equivalent for linear/no-KSCORE configs (cur_rb8). Set the CARE_KV
    # internal env too so both the ctor arg and the layer code agree.
    corr_impl = os.environ.get("CAREKV_CFG_CORR_IMPL", "vectorized")
    os.environ["CAREKV_CORRECTION_IMPL"] = corr_impl
    ad = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="blockgtq_style",
                       sk=sk, sv=sv, rk=rk, rv=rv,
                       k_channel_group=kcg, v_token_block=vtb,
                       k_store_mode="post_rope", correction_impl=corr_impl)
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

    # Record the residual config in BOTH the log and the CSV. Without this a
    # ladder CSV silently mixes configs (fine_rb8 vs cur_rb8 give Δ% differing
    # by 0.2-0.4pp), which reads as false model-dependence — see the
    # Llama-2-13b row in improved_ladder.csv, whose config is unrecoverable.
    cfg = dict(kcg=os.environ.get("CAREKV_CFG_KCG", "32"),
               vtb=os.environ.get("CAREKV_CFG_VTB", "4"),
               kmode=os.environ.get("CAREKV_CFG_KMODE", "linear"),
               rb=os.environ.get("CAREKV_CFG_RB", "4"),
               sk=os.environ.get("CAREKV_CFG_SK", "2"),
               sv=os.environ.get("CAREKV_CFG_SV", "4"),
               rk=os.environ.get("CAREKV_CFG_RK", "2"),
               rv=os.environ.get("CAREKV_CFG_RV", "2"))
    corr_impl = os.environ.get("CAREKV_CFG_CORR_IMPL", "vectorized")
    cfg_str = "kcg{kcg}_vtb{vtb}_{kmode}_rb{rb}_s{sk}{sv}_r{rk}{rv}".format(**cfg) + f"_{corr_impl}"
    cfg_name = os.environ.get("CAREKV_CFG_NAME", cfg_str)
    # n_calib is env-tunable purely to make a fast diagnostic PROBE (a smaller
    # calib changes B/C ppl, so a reduced value is throwaway-only, never paper).
    n_calib = int(os.environ.get("CAREKV_CFG_NCALIB", "2048"))

    print(f"[bgtq-adapter] {a.model_id} SL={a.seq_len} N={a.num_samples}", flush=True)
    print(f"[bgtq-adapter] config={cfg_name} [{cfg_str}] n_calib={n_calib}", flush=True)
    tc = time.perf_counter()
    meta = calibrate(a.model_id, n_calib=n_calib)
    print(f"[bgtq-adapter] calibrated {meta} in {time.perf_counter()-tc:.1f}s", flush=True)

    import gc
    csk = int(os.environ.get("CAREKV_CFG_SK", "2")); csv_ = int(os.environ.get("CAREKV_CFG_SV", "4"))
    crk = int(os.environ.get("CAREKV_CFG_RK", "2")); crv = int(os.environ.get("CAREKV_CFG_RV", "2"))
    B = one_arm(a.model_id, a.seq_len, a.num_samples, "B base_quant", 0, 0, 0, 0)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    C = one_arm(a.model_id, a.seq_len, a.num_samples, "C carekv", csk, csv_, crk, crv)
    delta = (C.ppl - B.ppl) if (B.ppl and C.ppl) else None

    row = dict(model=a.model_id, seq_len=a.seq_len, num_samples=a.num_samples,
               config=cfg_name, config_full=cfg_str,
               B_ppl=B.ppl, C_ppl=C.ppl, delta=round(delta, 4) if delta is not None else "",
               delta_pct=round(100*delta/B.ppl, 3) if delta is not None and B.ppl else "",
               C_k_reads=C.k_reads, C_v_reads=C.v_reads,
               B_k_reads=B.k_reads, B_v_reads=B.v_reads,
               B_runtime_s=B.runtime_seconds, C_runtime_s=C.runtime_seconds)
    os.makedirs(os.path.dirname(a.append_csv) or ".", exist_ok=True)
    wh = not os.path.exists(a.append_csv) or os.path.getsize(a.append_csv) == 0
    if not wh:
        # Appending a row whose fields don't match the existing header would
        # silently shift every column. Refuse instead — the run is already done,
        # so the result is printed below and nothing is lost.
        with open(a.append_csv, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != list(row.keys()):
            print(f"[bgtq-adapter] ERROR: header mismatch in {a.append_csv}\n"
                  f"  file: {existing}\n  row : {list(row.keys())}\n"
                  f"  NOT appending — write to a fresh CSV instead.", flush=True)
            print(f"[bgtq-adapter] DONE  B={B.ppl}  C={C.ppl}  Δ={row['delta']} "
                  f"({row['delta_pct']}%)  C_reads K={C.k_reads}/V={C.v_reads}", flush=True)
            return
    with open(a.append_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if wh: w.writeheader()
        w.writerow(row)
    print(f"[bgtq-adapter] DONE  B={B.ppl}  C={C.ppl}  Δ={row['delta']} ({row['delta_pct']}%)  "
          f"C_reads K={C.k_reads}/V={C.v_reads}", flush=True)


if __name__ == "__main__":
    main()
