"""tools/eval_rotation_confirm.py — Phase: rotation-CARE-KV confirm stage.

Screening (TinyLlama, SL128, N4) returned GO for **Hadamard pre-RoPE + CARE-KV**
(rot_pre_carekv 13.259 < uniform_carekv 13.462, complementary). This confirms the
winner at scale: 4 fair-INT3 models × {256,512,1024} × N≥8, with the same run_one /
KVMethodAdapter harness so numbers are directly comparable, INCLUDING TurboQuant
(a KVMethodAdapter) for the gap-flip test.

Arms (per model × seq_len): fp16, base_int3, uniform_carekv, rot_pre_carekv, turbo.
Success criterion (design doc): rot_pre_carekv flips ≥3 of CARE-KV's TurboQuant losses
(esp. DeepSeek / long context) WITHOUT regressing where CARE-KV already matched/won,
and beats uniform_carekv (the bar) with no KV-memory increase.

Resumable (skips done model,seq_len,arm). Per-cell exception isolation.

Run:
  CUDA_VISIBLE_DEVICES=5 python tools/eval_rotation_confirm.py \
    --out_csv results/rotation_confirm/rotation_confirm_n8.csv \
    --num-samples 8 --seq-lens 512 1024 256 \
    --models deepseek-ai/deepseek-llm-7b-base 01-ai/Yi-6B \
             mistralai/Mistral-7B-v0.3 openlm-research/open_llama_7b_v2
"""
import os, sys, csv, time, math, argparse
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/CARE_KV/care_kv/tools")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
# Use the production vectorized residual path (≈10× faster than the cached prototype
# loop) so the 7B×N8 grid is feasible. setdefault → overridable; the CAREKVAdapter
# does NOT set these, so they persist into the layer.py gate.
os.environ.setdefault("CAREKV_VECTORIZED_RESIDUAL", "1")
os.environ.setdefault("CAREKV_VECTORIZE_VDOM_ONLY", "1")

import torch
from CARE_KV.care_kv.baselines import (KVMethodAdapter, FP16Adapter, BaseQuantAdapter, CAREKVAdapter)
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter
from eval_base_quantizer_expansion import run_one


def maxp_for(sl):
    return max(16, math.ceil(sl / 16) + 8)


def carekv(bq, kmode, maxp):
    return CAREKVAdapter(mode="fixed", bits=3, base_quantizer=bq, k_store_mode=kmode,
                         bits_k=3, bits_v=3, sk=2, sv=4, rk=2, rv=2, max_pages=maxp)


# arm -> factory(maxp)
def arm_factory(arm, maxp):
    if arm == "fp16":           return FP16Adapter()
    if arm == "base_int3":      return BaseQuantAdapter(bits=3)
    if arm == "uniform_carekv": return carekv("uniform", "post_rope", maxp)
    if arm == "rot_pre_carekv": return carekv("rotatekv_style", "pre_rope", maxp)
    if arm == "turbo":          return TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=0, use_qjl=True)
    raise ValueError(arm)


ARMS = ["fp16", "base_int3", "uniform_carekv", "rot_pre_carekv", "turbo"]
COLS = ["model_id", "seq_len", "num_samples", "arm", "method_name", "ppl",
        "dppl_vs_fp16", "dppl_vs_base", "dppl_vs_uniform_carekv", "dppl_vs_turbo",
        "k_reads", "v_reads", "residual_memory_MB", "estimated_kv_memory_MB",
        "runtime_s", "status", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 256])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=ARMS)
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)

    done = set()
    rows = []
    if os.path.exists(A.out_csv):
        rows = list(csv.DictReader(open(A.out_csv)))
        done = {(r["model_id"], r["seq_len"], r["arm"]) for r in rows}
        print(f"[rot-confirm] resume: {len(done)} cells done", flush=True)

    def flush():
        with open(A.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore"); w.writeheader()
            for r in rows: w.writerow(r)

    for mid in A.models:
        for sl in A.seq_lens:
            maxp = maxp_for(sl)
            cell_ppl = {}   # arm -> ppl for deltas within (model, sl)
            # pre-load any already-done ppls for delta computation
            for r in rows:
                if r["model_id"] == mid and r["seq_len"] == str(sl):
                    try: cell_ppl[r["arm"]] = float(r["ppl"])
                    except Exception: pass
            for arm in A.arms:
                if (mid, str(sl), arm) in done:
                    print(f"[rot-confirm] skip {mid} SL{sl} {arm} (resume)", flush=True); continue
                t0 = time.perf_counter()
                try:
                    ad = arm_factory(arm, maxp)
                    r = run_one(ad, mid, "wikitext", sl, A.num_samples)
                    ppl = float(r.ppl) if (r.ppl and math.isfinite(float(r.ppl))) else float("nan")
                    rec = dict(model_id=mid, seq_len=sl, num_samples=A.num_samples, arm=arm,
                               method_name=r.method_name, ppl=round(ppl, 4) if math.isfinite(ppl) else "nan",
                               k_reads=r.k_reads, v_reads=r.v_reads,
                               residual_memory_MB=round(r.residual_memory_MB, 3),
                               estimated_kv_memory_MB=round(getattr(r, "estimated_kv_memory_MB", 0) or 0, 2),
                               runtime_s=round(time.perf_counter() - t0, 1),
                               status="real" if math.isfinite(ppl) else "nonfinite", notes="")
                    if math.isfinite(ppl):
                        cell_ppl[arm] = ppl
                    print(f"[rot-confirm] {mid.split('/')[-1]:22s} SL{sl} {arm:16s} "
                          f"PPL={rec['ppl']}  K={r.k_reads} V={r.v_reads}  ({rec['runtime_s']}s)", flush=True)
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    rec = dict(model_id=mid, seq_len=sl, num_samples=A.num_samples, arm=arm,
                               status="oom", notes=str(e)[:80], runtime_s=round(time.perf_counter()-t0, 1))
                    print(f"[rot-confirm] {mid} SL{sl} {arm} OOM", flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    rec = dict(model_id=mid, seq_len=sl, num_samples=A.num_samples, arm=arm,
                               status="error", notes=f"{type(e).__name__}: {str(e)[:80]}",
                               runtime_s=round(time.perf_counter()-t0, 1))
                rows.append(rec); flush()
            # fill deltas for this (model, sl)
            fp = cell_ppl.get("fp16"); bq = cell_ppl.get("base_int3")
            uni = cell_ppl.get("uniform_carekv"); tq = cell_ppl.get("turbo")
            for r in rows:
                if r["model_id"] == mid and r["seq_len"] == str(sl) and r.get("ppl") not in ("", "nan", None):
                    try: p = float(r["ppl"])
                    except Exception: continue
                    r["dppl_vs_fp16"] = round(p - fp, 4) if fp else ""
                    r["dppl_vs_base"] = round(p - bq, 4) if bq else ""
                    r["dppl_vs_uniform_carekv"] = round(p - uni, 4) if uni else ""
                    r["dppl_vs_turbo"] = round(p - tq, 4) if tq else ""
            flush()
    print(f"[rot-confirm] done. {len(rows)} rows -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
