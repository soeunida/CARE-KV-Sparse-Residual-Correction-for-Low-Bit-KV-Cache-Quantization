"""tools/eval_kv_residual_memory_scaling.py — Part C.

Analytical KV-memory vs CARE-KV-residual-memory scaling, to answer:
  1. At what batch/context does the KV cache reach GB scale?
  2. Does residual memory dominate or stay small?
  3. Does CARE-KV still save memory after residuals?
  4. Which store budget has the best PPL/memory trade-off?
  5. Does BaseQuant INT4 already solve the problem at this scale?

This is a *memory estimator* (no model forward) so it can sweep the full
batch×seq×bits×budget grid cheaply. The per-component model is grounded in
the project's measured `memory/memory_table.csv` decomposition
(base_code / scale / residual / meta / sketch), re-derived as per-slot and
per-token byte constants below so the numbers match the accepted model.

PPL is NOT recomputed here (that is Parts A/B/D); this part is pure memory.
"""
from __future__ import annotations
import argparse, csv, os

# ── Memory constants derived from memory/memory_table.csv (packed_base path) ──
# fp16 KV per (layer, kv_head, token, K+V) = 2 (K&V) * head_dim * 2 bytes.
# Packed base INT-b code: head_dim * b/8 bytes per (layer,kv_head,token) per K/V.
# int8 page scale: ~1/group_size byte/element (group_size=32) → negligible per-token term.
# Residual slot: ~1.13 bytes per stored slot (≈4-bit value + ~4-bit local index),
#   measured: 1.1667 MB / (22*4*2048*(2+4)) slots @ SK2SV4 SL2048.
# Meta (residual index/bookkeeping): ~0.524 bytes per stored slot.
# Router sketch: sketch_dim int8 per (layer,kv_head,token) ≈ 4.0 bytes/token @ dim16.
RESIDUAL_BYTES_PER_SLOT = 1.13
META_BYTES_PER_SLOT = 0.524
SCALE_BYTES_PER_ELEM = 1.0 / 32.0      # int8 scale per group of 32
SKETCH_BYTES_PER_TOKEN_PER_LAYER_KVHEAD = 4.0   # sketch_dim=16 int8 ≈ 4 B (router)

GB = 1024 ** 3
MB = 1024 ** 2

MODELS = {
    # name: (num_layers, num_kv_heads, head_dim, num_attn_heads)
    "Qwen2.5-7B": (28, 4, 128, 28),
    "TinyLlama-1.1B": (22, 4, 64, 32),
    # hypothetical MHA-7B (no GQA) to show the GQA memory win at scale
    "LLaMA-7B-MHA(proj)": (32, 32, 128, 32),
}

BUDGETS = [
    # (SK, SV, RK, RV)
    (1, 2, 1, 1),
    (2, 4, 2, 2),    # paper-best
    (4, 4, 2, 2),
]


def fp16_kv_bytes(L, Hkv, Dh, B, S):
    return L * 2 * B * S * Hkv * Dh * 2


def estimate(L, Hkv, Dh, B, S, bits, sk, sv, with_residual=True):
    tokens = B * S
    elems = L * Hkv * tokens * Dh                       # per K (or V)
    fp16 = fp16_kv_bytes(L, Hkv, Dh, B, S)
    base_code = 2 * elems * (bits / 8.0)                # K+V packed b-bit
    scale = 2 * elems * SCALE_BYTES_PER_ELEM            # int8 page scales
    if with_residual:
        stored_slots = L * Hkv * tokens * (sk + sv)
        residual = stored_slots * RESIDUAL_BYTES_PER_SLOT
        meta = stored_slots * META_BYTES_PER_SLOT
        sketch = L * Hkv * tokens * SKETCH_BYTES_PER_TOKEN_PER_LAYER_KVHEAD
    else:
        residual = meta = sketch = 0.0
    total = base_code + scale + residual + meta + sketch
    return dict(fp16=fp16, base_code=base_code, scale=scale,
                residual=residual, meta=meta, sketch=sketch, total=total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--models", default="Qwen2.5-7B,TinyLlama-1.1B,LLaMA-7B-MHA(proj)")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--seqlens", default="128,512,1024,2048")
    ap.add_argument("--bits", default="4,3,2")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    seqlens = [int(x) for x in args.seqlens.split(",")]
    bitlist = [int(x) for x in args.bits.split(",")]

    rows = []
    for mname in models:
        L, Hkv, Dh, Hq = MODELS[mname]
        for B in batches:
            for S in seqlens:
                fp16 = fp16_kv_bytes(L, Hkv, Dh, B, S)
                for bits in bitlist:
                    # BaseQuant (no residual)
                    bq = estimate(L, Hkv, Dh, B, S, bits, 0, 0, with_residual=False)
                    rows.append(dict(
                        model=mname, gqa=(Hkv < Hq), num_layers=L, kv_heads=Hkv,
                        head_dim=Dh, batch=B, seq_len=S, tokens=B * S, bits=bits,
                        method=f"BaseQuant_INT{bits}", store_budget="-",
                        fp16_kv_GB=round(fp16 / GB, 6),
                        base_kv_GB=round(bq["base_code"] / GB, 6),
                        residual_GB=0.0, metadata_GB=round(bq["scale"] / GB, 6),
                        total_kv_GB=round(bq["total"] / GB, 6),
                        memory_saving_ratio=round(bq["total"] / fp16, 4),
                        residual_frac_of_total=0.0,
                        reaches_GB_scale=(fp16 >= GB),
                    ))
                    # CARE-KV at each budget
                    for (sk, sv, rk, rv) in BUDGETS:
                        e = estimate(L, Hkv, Dh, B, S, bits, sk, sv, with_residual=True)
                        rows.append(dict(
                            model=mname, gqa=(Hkv < Hq), num_layers=L, kv_heads=Hkv,
                            head_dim=Dh, batch=B, seq_len=S, tokens=B * S, bits=bits,
                            method=f"CAREKV_INT{bits}", store_budget=f"SK{sk}SV{sv}RK{rk}RV{rv}",
                            fp16_kv_GB=round(fp16 / GB, 6),
                            base_kv_GB=round((e["base_code"] + e["scale"]) / GB, 6),
                            residual_GB=round(e["residual"] / GB, 6),
                            metadata_GB=round((e["meta"] + e["sketch"]) / GB, 6),
                            total_kv_GB=round(e["total"] / GB, 6),
                            memory_saving_ratio=round(e["total"] / fp16, 4),
                            residual_frac_of_total=round(
                                (e["residual"]) / max(e["total"], 1e-12), 4),
                            reaches_GB_scale=(fp16 >= GB),
                        ))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows -> {args.out_csv}")

    # Quick headline findings to stdout.
    q = [r for r in rows if r["model"] == "Qwen2.5-7B"]
    gb_corner = max(q, key=lambda r: r["tokens"])
    print(f"\nQwen2.5-7B (GQA, 4 kv-heads) top corner B={gb_corner['batch']} "
          f"S={gb_corner['seq_len']}: fp16 KV = {gb_corner['fp16_kv_GB']:.3f} GB")
    pb = [r for r in q if r["method"] == "CAREKV_INT3" and r["store_budget"] == "SK2SV4RK2RV2"
          and r["batch"] == gb_corner["batch"] and r["seq_len"] == gb_corner["seq_len"]][0]
    print(f"  CARE-KV INT3 SK2SV4: total {pb['total_kv_GB']:.4f} GB "
          f"({pb['memory_saving_ratio']:.3f}x fp16), residual = "
          f"{pb['residual_frac_of_total']*100:.1f}% of total")


if __name__ == "__main__":
    main()
