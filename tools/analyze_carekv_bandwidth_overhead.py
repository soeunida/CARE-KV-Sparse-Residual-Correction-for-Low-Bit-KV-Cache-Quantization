"""tools/analyze_carekv_bandwidth_overhead.py

Theoretical FLOPs + MEMORY-BANDWIDTH overhead of CARE-KV's decode-time residual
correction, relative to base INT3 attention. Extends the FLOP model in
profile_carekv_decode_overhead.py with an HBM-traffic (bandwidth) model, since
KV-cache attention is bandwidth-bound and a fused kernel's benefit is a
bandwidth question, not a FLOP question.

Per decode token, summed over L layers, context length T. Config = paper-best
(INT3 base, 4-bit residual, int8 scales, SK2 SV4 RK2 RV2, sketch_dim=32,
page=16, k_channel_group=32, v_token_block=4).

Base attention:
  FLOP  = L · 4 · Hq · T · Dh                      (QK^T + AV, 2 MAC each)
  bytes = L · Hkv · 2·T·(Dh·b_base + Dh/gs·b_scale) (read whole K+V cache/token)
Correction (residual read once per kv_head — cached/shared across the GQA group):
  FLOP  = L · 2·(RK·Hq·Dh + RV·Hq·Dh + (SK+SV)·sketch·Hq)
  bytes = L · Hkv · (RK·page·kcg·b_res + RV·vtb·Dh·b_res + (RK+RV)·b_scale2)
Also reports an "applied" (per-query, no sharing) upper-bound residual byte count.

Roofline: arithmetic intensity FLOP/byte vs the accelerator ridge point
(A6000 ~ 150 TFLOP fp16 / 1.5 TB/s ≈ 100 FLOP/B) → both base and correction are
bandwidth-bound; the correction adds a CONTEXT-INDEPENDENT byte cost, so its
bandwidth overhead shrinks ~1/T and is <2% in the long-context regime that
matters for KV compression.

Outputs: <out-csv> + <out-md>.
"""
from __future__ import annotations
import argparse, csv, math, os

B_BASE = 3 / 8      # INT3 base code byte/value
B_RES = 0.5         # 4-bit residual byte/value
B_SCALE = 1         # int8 scale byte/value (per group)
GS = 32             # group_size
PAGE, KCG, VTB = 16, 32, 4
SK, SV, RK, RV, SKETCH = 2, 4, 2, 2, 32
RIDGE = 100.0       # FLOP/byte ridge point (A6000-class: ~150 TFLOP fp16 / 1.5 TB/s)

CONFIGS = {
    "TinyLlama-1.1B": dict(L=22, Hq=32, Hkv=4, Dh=64),
    "Mistral-7B (GQA)": dict(L=32, Hq=32, Hkv=8, Dh=128),
    "DeepSeek-7B (MHA)": dict(L=30, Hq=32, Hkv=32, Dh=128),
}
SEQLENS = [128, 512, 1024, 2048, 4096, 8192]


def row(name, L, Hq, Hkv, Dh, T):
    # ── FLOP (per decode token) ──
    base_flop = L * 4 * Hq * T * Dh
    corr_flop = L * 2 * (RK * Hq * Dh + RV * Hq * Dh + (SK + SV) * SKETCH * Hq)
    # ── bytes (per decode token) ──
    base_bytes = L * Hkv * 2 * T * (Dh * B_BASE + (Dh / GS) * B_SCALE)
    corr_bytes_shared = L * Hkv * (RK * PAGE * KCG * B_RES + RV * VTB * Dh * B_RES
                                   + (RK + RV) * 2)     # +fp16 scale/slot
    corr_bytes_applied = L * Hq * (RK + RV) * Dh * B_RES  # per-query upper bound
    return dict(
        model=name, seq_len=T,
        base_GFLOP=round(base_flop / 1e9, 4),
        corr_GFLOP=round(corr_flop / 1e9, 4),
        flop_overhead_pct=round(100 * corr_flop / base_flop, 3),
        base_KB=round(base_bytes / 1024, 1),
        corr_KB_shared=round(corr_bytes_shared / 1024, 2),
        corr_KB_applied=round(corr_bytes_applied / 1024, 2),
        bw_overhead_shared_pct=round(100 * corr_bytes_shared / base_bytes, 3),
        bw_overhead_applied_pct=round(100 * corr_bytes_applied / base_bytes, 3),
        base_AI=round(base_flop / base_bytes, 1),
        corr_AI=round(corr_flop / corr_bytes_shared, 1),
        base_bound=("BW" if base_flop / base_bytes < RIDGE else "compute"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", default="results/router_diagnostic/carekv_overhead_analysis.csv")
    ap.add_argument("--out-md", default="results/router_diagnostic/carekv_overhead_analysis.md")
    args = ap.parse_args()

    rows = []
    for name, c in CONFIGS.items():
        for T in SEQLENS:
            rows.append(row(name, T=T, **c))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)
    print("wrote", args.out_csv)

    L = []
    L.append("# CARE-KV correction overhead — theoretical FLOPs + memory bandwidth")
    L.append("")
    L.append("Per decode token, summed over layers. Base = INT3 attention (read "
             "whole K+V cache/token); correction = residual read + apply + router "
             "(paper-best SK2 SV4 RK2 RV2, 4-bit residual, sketch_dim=32). "
             "`shared` = residual read once per KV head (cached, GQA-shared); "
             "`applied` = per-query upper bound. Ridge point "
             f"≈{RIDGE:.0f} FLOP/byte (A6000-class).")
    L.append("")
    for name in CONFIGS:
        sub = [r for r in rows if r["model"] == name]
        L.append(f"## {name}")
        L.append("")
        L.append("| SL | base GFLOP | base KB | corr FLOP% | corr KB (shared) | "
                 "**BW overhead (shared)** | BW overhead (applied) | base AI | bound |")
        L.append("|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
        for r in sub:
            L.append(f"| {r['seq_len']} | {r['base_GFLOP']} | {r['base_KB']} | "
                     f"{r['flop_overhead_pct']}% | {r['corr_KB_shared']} | "
                     f"**{r['bw_overhead_shared_pct']}%** | {r['bw_overhead_applied_pct']}% | "
                     f"{r['base_AI']} | {r['base_bound']} |")
        L.append("")
    L.append("## Reading")
    L.append("")
    t = {(r['model'], r['seq_len']): r for r in rows}
    tl = "TinyLlama-1.1B"
    L.append(f"- **FLOP overhead is tiny** ({t[(tl,128)]['flop_overhead_pct']}% at "
             f"SL128 → {t[(tl,2048)]['flop_overhead_pct']}% at SL2048) — correction "
             "is negligible arithmetic.")
    L.append(f"- **Bandwidth is the real axis**, and it too is small in the "
             f"long-context regime: shared-read overhead "
             f"{t[(tl,128)]['bw_overhead_shared_pct']}% (SL128) → "
             f"{t[(tl,1024)]['bw_overhead_shared_pct']}% (SL1024) → "
             f"{t[(tl,8192)]['bw_overhead_shared_pct']}% (SL8192). The residual "
             "read is **context-independent** (fixed budget/token), so its share "
             "of the ∝T base KV read shrinks ~1/T.")
    L.append(f"- **Both base and correction are bandwidth-bound** (base AI "
             f"≈{t[(tl,1024)]['base_AI']} ≪ ridge {RIDGE:.0f} FLOP/byte). So the "
             "cost that matters is HBM traffic, and the correction adds <2% of it "
             "at SL≥1024.")
    L.append("- **GQA amortizes correction bandwidth**: fewer KV heads → the "
             "shared residual read is smaller relative to the (Hq-driven) base "
             "compute; MHA (DeepSeek) has proportionally more KV-head residual "
             "reads but the overhead is still small at long context.")
    L.append("- **Conclusion.** The correction's FLOP and bandwidth overheads are "
             "both **<2% at deployment-relevant context lengths**; the ~1000× "
             "prototype slowdown is entirely the per-token Python loop, not the "
             "algorithm. A fused gather+dequant+apply kernel would realize this "
             "sub-2% theoretical overhead; the vectorized path already recovers "
             "most of it (~15–80× measured).")
    L.append("")
    L.append("**Status: analytical** (arithmetic counts; measured walltime in "
             "`carekv_decode_overhead.csv`).")
    open(args.out_md, "w").write("\n".join(L) + "\n")
    print("wrote", args.out_md)


if __name__ == "__main__":
    main()
