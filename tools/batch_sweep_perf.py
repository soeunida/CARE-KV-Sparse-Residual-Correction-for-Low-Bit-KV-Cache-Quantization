"""batch_sweep_perf.py — CARE-KV paper-best: performance vs batch size.

Single model instance (paper-best carekv_stored, INT3, joint+both), swept over
batch sizes. For each batch B measures, on WikiText-2 test windows (uniform
length SL, so batching needs NO padding):

  - PPL                (quality; must be ~batch-invariant)
  - forward latency    (ms per [B, SL] forward, use_cache=False — the eval path)
  - throughput         (tok/s = B*SL / latency)
  - peak GPU memory    (MB)
  - K_reads / V_reads  (router-fired sanity, per CLAUDE.md)

Outputs CSV + PNG. TinyLlama by default (paper-best MODEL_ID).

Env: MODEL_ID, SEQ_LEN(=256), NUM_SAMPLES(=16), BATCHES("1 2 4 8"),
     CALIB_TOKENS(=2048), SEED(=0), WARMUP(=1), OUT_DIR.
"""
from __future__ import annotations
import os, sys, math, csv, time
import torch

sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/blockgtq")
sys.path.insert(0, "/home/soeun/care_kv_clean")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer, LlamaForCausalLM
from run_blockgtq_carekv import _wt2_text, _build_carekv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "256"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "16"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "1 2 4 8").split()]
CALIB_TOKENS = int(os.environ.get("CALIB_TOKENS", "2048"))
SEED = int(os.environ.get("SEED", "0"))
WARMUP = int(os.environ.get("WARMUP", "1"))
RESIDUAL_ON = os.environ.get("RESIDUAL_ON", "1") == "1"   # 1=carekv, 0=base_quant
MODE_TAG = "carekv" if RESIDUAL_ON else "base_quant"
OUT_DIR = os.environ.get("OUT_DIR", "results/batch_sweep")


def reset_carekv(model):
    for sub in model.modules():
        if hasattr(sub, "reset_cache") and hasattr(sub, "_caches"):
            sub.reset_cache()


def eval_batch(model, chunks, B):
    """PPL + latency/throughput/peak-mem for one batch size over all windows."""
    from CARE_KV.care_kv import get_debug_stats, reset_debug_stats
    reset_debug_stats()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    N = chunks.shape[0]
    # ---- warmup (excluded from timing) ----
    with torch.no_grad():
        for _ in range(WARMUP):
            ids = chunks[:B].to(DEVICE)
            model(input_ids=ids, labels=ids, use_cache=False)
            reset_carekv(model)
    torch.cuda.synchronize()
    # ---- timed pass ----
    total_loss, total_tok, fwd_ms = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, N, B):
            ids = chunks[i:i + B].to(DEVICE)
            b = ids.shape[0]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model(input_ids=ids, labels=ids, use_cache=False)
            torch.cuda.synchronize()
            fwd_ms.append((time.perf_counter() - t0) * 1e3)
            n = b * (SEQ_LEN - 1)
            total_loss += float(out.loss.item()) * n
            total_tok += n
            reset_carekv(model)
    ppl = math.exp(total_loss / total_tok)
    peak = torch.cuda.max_memory_allocated() / 1e6
    st = get_debug_stats()
    # mean per-forward latency; throughput on FULL batch of tokens processed.
    import statistics
    lat_ms = statistics.mean(fwd_ms)
    # tokens processed per forward = B*SL (all positions), throughput in tok/s.
    thr = (B * SEQ_LEN) / (lat_ms / 1e3)
    return dict(
        batch=B, ppl=round(ppl, 6),
        fwd_ms_mean=round(lat_ms, 2),
        fwd_ms_per_seq=round(lat_ms / B, 2),
        throughput_tok_s=round(thr, 1),
        peak_gpu_mem_MB=round(peak, 1),
        peak_mem_per_seq_MB=round(peak / B, 1),
        n_forwards=len(fwd_ms),
        K_reads=st.get("k_slots_read", 0), V_reads=st.get("v_slots_read", 0),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    # eval windows: WT2 test, uniform length SL
    full = "\n\n".join(_wt2_text("test"))
    ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    need = SEQ_LEN * NUM_SAMPLES
    assert ids.numel() >= need, f"WT2 too short: {ids.numel()} < {need}"
    chunks = ids[:need].view(NUM_SAMPLES, SEQ_LEN)
    print(f"[batch-sweep] {MODEL_ID} SL={SEQ_LEN} N={NUM_SAMPLES} "
          f"batches={BATCHES} device={DEVICE}", flush=True)

    # build model + calibrate blockgtq once, then carekv paper-best (residual on)
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None).eval()
    model.config.use_cache = False
    import CARE_KV.care_kv.blockgtq_base as bgb
    ctext = "\n\n".join(_wt2_text("train"))
    calib_ids = tok(ctext, return_tensors="pt",
                    add_special_tokens=False)["input_ids"][:, :CALIB_TOKENS]
    bgb.reset()
    bgb.calibrate(model, calib_ids, k_avg_bits=3, v_bits=3, device=DEVICE,
                  n_calib_tokens=CALIB_TOKENS)
    model = _build_carekv(model, base_bits=3, residual_on=RESIDUAL_ON)
    print(f"[batch-sweep] calibrated + {MODE_TAG} built "
          f"(residual_on={RESIDUAL_ON})", flush=True)

    rows = []
    for B in BATCHES:
        if NUM_SAMPLES % B != 0:
            print(f"[batch-sweep] skip B={B}: N={NUM_SAMPLES} not divisible", flush=True)
            continue
        try:
            r = eval_batch(model, chunks, B)
            rows.append(r)
            print(f"[batch-sweep] B={B:<3} PPL={r['ppl']:.4f} "
                  f"fwd={r['fwd_ms_mean']}ms ({r['fwd_ms_per_seq']}ms/seq) "
                  f"thr={r['throughput_tok_s']}tok/s peak={r['peak_gpu_mem_MB']}MB "
                  f"K/V_reads={r['K_reads']}/{r['V_reads']}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[batch-sweep] B={B} FAILED: {type(e).__name__}: {e}", flush=True)

    if not rows:
        print("[batch-sweep] no rows produced"); return
    csv_path = os.path.join(OUT_DIR, "batch_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[batch-sweep] wrote {csv_path}", flush=True)
    make_plot(rows, os.path.join(OUT_DIR, "batch_sweep.png"))


def make_plot(rows, png_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = [r["batch"] for r in rows]
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"CARE-KV {MODE_TAG} — performance vs batch size\n"
                 f"{MODEL_ID}  SL={SEQ_LEN}  N={NUM_SAMPLES}", fontsize=12)
    ax[0, 0].plot(B, [r["throughput_tok_s"] for r in rows], "o-", color="#2a9d8f")
    ax[0, 0].set(title="Throughput", xlabel="batch size", ylabel="tok/s")
    ax[0, 1].plot(B, [r["fwd_ms_per_seq"] for r in rows], "o-", color="#e76f51")
    ax[0, 1].set(title="Latency per sequence", xlabel="batch size", ylabel="ms / seq")
    ax[1, 0].plot(B, [r["peak_gpu_mem_MB"] for r in rows], "o-", color="#e9c46a")
    ax[1, 0].plot(B, [r["peak_mem_per_seq_MB"] for r in rows], "s--",
                  color="#f4a261", label="per seq")
    ax[1, 0].set(title="Peak GPU memory", xlabel="batch size", ylabel="MB")
    ax[1, 0].legend()
    ax[1, 1].plot(B, [r["ppl"] for r in rows], "o-", color="#264653")
    ax[1, 1].set(title="PPL (quality — should be ~flat)",
                 xlabel="batch size", ylabel="WikiText-2 PPL")
    for a in ax.flat:
        a.grid(alpha=0.3); a.set_xticks(B)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(png_path, dpi=130)
    print(f"[batch-sweep] wrote {png_path}", flush=True)


if __name__ == "__main__":
    main()
