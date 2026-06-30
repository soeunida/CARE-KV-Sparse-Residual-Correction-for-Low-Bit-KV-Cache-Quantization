"""tools/eval_gqa_scaling.py — GQA CARE-KV scaling across model sizes (T8).

CARE-KV vs BaseQuant_INT3 vs TurboQuant_INT3 on a GQA model ladder spanning sizes:
TinyLlama-1.1B (GQA4) → Yi-6B (GQA4) → Mistral-7B (GQA8) → SOLAR-10.7B (GQA8).
Llama/Mistral-family (loadable as LlamaForCausalLM). Uniform-packed INT3 base (the
production-stable path), correction_impl=vectorized. Per model: GQA config (KV heads /
groups / head_dim), PPL, Δ vs Base/Turbo, collapse flag, K/V reads.

(Replaces the unrecovered final_*_scaling CSVs with a fresh GQA-focused results table.)

Run:
  CUDA_VISIBLE_DEVICES=2 python tools/eval_gqa_scaling.py \
    --out_csv results/gqa_scaling/gqa_scaling.csv --num-samples 4 --seq-len 512 \
    --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 01-ai/Yi-6B mistralai/Mistral-7B-v0.3 upstage/SOLAR-10.7B-v1.0
"""
import os, sys, csv, time, math, argparse
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/CARE_KV/care_kv/tools")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoConfig
from CARE_KV.care_kv.baselines import FP16Adapter, BaseQuantAdapter, CAREKVAdapter
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from eval_base_quantizer_expansion import run_one


def maxp_for(sl):
    return max(16, math.ceil(sl / 16) + 8)


def adapter_for(arm, maxp):
    if arm == "fp16":           return FP16Adapter()
    if arm == "base_int3":      return BaseQuantAdapter(bits=3)
    if arm == "carekv_SK2SV4":  return CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                                                     k_store_mode="post_rope", bits_k=3, bits_v=3,
                                                     sk=2, sv=4, rk=2, rv=2, max_pages=maxp,
                                                     correction_impl="vectorized")
    if arm == "turbo_int3":     return TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=0, use_qjl=True)
    raise ValueError(arm)


ARMS = ["fp16", "base_int3", "carekv_SK2SV4", "turbo_int3"]
COLS = ["model_id", "model_type", "params_class", "num_layers", "num_heads", "num_kv_heads",
        "kv_groups", "head_dim", "seq_len", "num_samples", "arm", "ppl",
        "delta_vs_fp16", "delta_vs_basequant_int3", "delta_vs_turboquant_int3",
        "k_reads", "v_reads", "collapse", "runtime_s", "status", "notes"]


def fv(x):
    try: return float(x)
    except Exception: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--models", nargs="+", required=True)
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)
    rows, done = [], set()
    if os.path.exists(A.out_csv):
        rows = list(csv.DictReader(open(A.out_csv)))
        done = {(r["model_id"], r["arm"]) for r in rows}

    def flush():
        with open(A.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
            for r in rows: w.writerow(r)

    print(f"[gqa] models={A.models} SL{A.seq_len} N{A.num_samples}", flush=True)
    for mid in A.models:
        cfg = AutoConfig.from_pretrained(mid)
        H = cfg.num_attention_heads
        KV = getattr(cfg, "num_key_value_heads", H)
        hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
        grp = H // KV if KV else 1
        gqa = {"model_type": cfg.model_type, "num_layers": cfg.num_hidden_layers,
               "num_heads": H, "num_kv_heads": KV, "kv_groups": grp, "head_dim": hd,
               "params_class": mid.split("/")[-1]}
        maxp = maxp_for(A.seq_len)
        ref = None
        for arm in ARMS:
            if (mid, arm) in done:
                print(f"[gqa] skip {mid} {arm}", flush=True); continue
            t0 = time.perf_counter()
            try:
                r = run_one(adapter_for(arm, maxp), mid, "wikitext", A.seq_len, A.num_samples)
                ppl = float(r.ppl) if (r.ppl and math.isfinite(float(r.ppl))) else float("nan")
                if arm == "fp16" and math.isfinite(ppl):
                    ref = ppl
                coll = (ref is not None and math.isfinite(ppl) and ppl > max(100.0, 5.0 * ref))
                rec = dict(model_id=mid, seq_len=A.seq_len, num_samples=A.num_samples, arm=arm,
                           ppl=round(ppl, 4) if math.isfinite(ppl) else "nan",
                           delta_vs_fp16=round(ppl - ref, 4) if (ref and math.isfinite(ppl)) else "",
                           k_reads=r.k_reads, v_reads=r.v_reads,
                           collapse="yes" if coll else "no",
                           runtime_s=round(time.perf_counter() - t0, 1),
                           status="unstable_outlier_collapse" if coll else "real", **gqa)
                print(f"[gqa] {mid.split('/')[-1]:24s} {arm:14s} PPL={rec['ppl']}  "
                      f"K={r.k_reads} V={r.v_reads}  coll={rec['collapse']}  ({rec['runtime_s']}s)", flush=True)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rec = dict(model_id=mid, arm=arm, status="oom", notes=str(e)[:60], **gqa)
                print(f"[gqa] {mid} {arm} OOM", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                rec = dict(model_id=mid, arm=arm, status="error",
                           notes=f"{type(e).__name__}: {str(e)[:70]}", **gqa)
            rows.append(rec); flush()
        # fill deltas vs base/turbo for this model
        def g(a):
            for r in rows:
                if r["model_id"] == mid and r["arm"] == a:
                    return fv(r.get("ppl"))
            return None
        bq, tq = g("base_int3"), g("turbo_int3")
        for r in rows:
            if r["model_id"] == mid and fv(r.get("ppl")) is not None:
                p = fv(r["ppl"])
                r["delta_vs_basequant_int3"] = round(p - bq, 4) if bq else ""
                r["delta_vs_turboquant_int3"] = round(p - tq, 4) if tq else ""
        flush()
    print(f"[gqa] done. {len(rows)} rows -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
