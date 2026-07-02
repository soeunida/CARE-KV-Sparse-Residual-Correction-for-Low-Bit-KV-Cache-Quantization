"""tools/eval_per_layer_importance.py — per-layer token-importance dispersion.

Deliverable #10: WHICH layers benefit most from query-aware routing. A layer whose
per-key attention mass is more DISPERSED (lower Gini / lower top-1% share / higher
entropy) has no single dominant key, so which residual slots to correct depends
strongly on the query -> query-aware routing helps that layer more. This dumps the
per-layer dispersion measured during a CARE-KV forward (CAREKV_DUMP_IMPORTANCE).

Run (needs a free GPU):
  CUDA_VISIBLE_DEVICES=6 python tools/eval_per_layer_importance.py \
    --model-id mistralai/Mistral-7B-v0.3 --seq-len 1024 --num-samples 2 \
    --out-csv results/longctx_ppl/per_layer_importance.csv \
    --out-fig results/longctx_ppl/fig_per_layer_importance.png
"""
from __future__ import annotations
import argparse, csv, os, sys
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CAREKV_DUMP_IMPORTANCE"] = "1"

import math, torch
from transformers import AutoTokenizer
from CARE_KV.care_kv.baselines import CAREKVAdapter
from CARE_KV.care_kv import get_imp_per_layer, reset_debug_stats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--out-csv", default="results/longctx_ppl/per_layer_importance.csv")
    ap.add_argument("--out-fig", default="results/longctx_ppl/fig_per_layer_importance.png")
    A = ap.parse_args()
    os.environ["CAREKV_CHUNKED_CORRECTION"] = "1"
    os.environ["CAREKV_CHUNK_SIZE"] = "512"

    tok = AutoTokenizer.from_pretrained(A.model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    from datasets import load_dataset
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    buf, tot = [], 0
    for ex in ds:
        t = ex.get("text", "")
        if t.strip():
            buf.append(t); tot += len(t)
        if tot >= A.seq_len * A.num_samples * 8 + 100000:
            break
    ids = tok("\n\n".join(buf), return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    windows = ids[: A.seq_len * A.num_samples].view(A.num_samples, A.seq_len)

    maxp = math.ceil(A.seq_len / 16) + 16
    ad = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                       k_store_mode="post_rope", bits_k=3, bits_v=3,
                       sk=2, sv=4, rk=2, rv=2, max_pages=maxp,
                       correction_impl="vectorized")
    reset_debug_stats()
    m = ad.setup_model(A.model_id)
    with torch.no_grad():
        for i in range(A.num_samples):
            x = windows[i:i+1].to(DEVICE)
            for sub in m.modules():
                if hasattr(sub, "reset_cache") and hasattr(sub, "_caches"):
                    sub.reset_cache()
            m(input_ids=x, use_cache=False)
    per = get_imp_per_layer()
    if not per:
        print("[per-layer] no data (router didn't fire?)"); return

    rows = [dict(layer=lid, **{k: round(v, 5) for k, v in d.items()})
            for lid, d in sorted(per.items())]
    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)
    with open(A.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[per-layer] wrote {A.out_csv} ({len(rows)} layers)")
    lids = [r["layer"] for r in rows]
    gini = [r["gini"] for r in rows]
    top1 = [r["top1pct_mass"] for r in rows]
    # report the most query-aware-favourable layers (most dispersed = lowest gini)
    disp = sorted(rows, key=lambda r: r["gini"])[:5]
    print("  most-dispersed layers (query-aware helps most):",
          [(r["layer"], r["gini"]) for r in disp])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(lids, gini, color="C2", alpha=0.8, label="Gini of per-key attn mass")
        ax.set_xlabel("layer index"); ax.set_ylabel("Gini (↓ = more dispersed)", color="C2")
        ax.tick_params(axis="y", labelcolor="C2")
        ax2 = ax.twinx()
        ax2.plot(lids, top1, "o-", color="C4", label="top-1% mass share")
        ax2.set_ylabel("top-1% mass share", color="C4"); ax2.tick_params(axis="y", labelcolor="C4")
        ax.set_title(f"Per-layer token-importance dispersion ({A.model_id.split('/')[-1]}, "
                     f"PG-19 SL={A.seq_len})\nmore-dispersed layers benefit most from query-aware routing")
        fig.tight_layout(); fig.savefig(A.out_fig, dpi=130, bbox_inches="tight")
        print(f"[per-layer] wrote {A.out_fig}")
    except Exception as e:
        print(f"[per-layer] fig skipped: {e}")


if __name__ == "__main__":
    main()
