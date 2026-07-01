"""tools/analyze_overhead_flops_bw.py — analytical FLOPs / memory-bandwidth
overhead model for CARE-KV decode.

Reviewer ask: quantify CARE-KV's overhead via a fused implementation OR, at
minimum, a theoretical FLOPs / memory-bandwidth analysis.

We give the analysis. It is the honest deliverable because CARE-KV's current
decode path is a per-(layer, kv_head, token) Python-loop PROTOTYPE, whose
wall-clock (documented in results/prefill_decode_perf/) is dominated by Python
interpreter overhead, NOT by the method's algorithmic cost. This model separates
the two: it reports the *algorithmic* extra FLOPs and extra bytes CARE-KV adds
per decode step, on top of plain INT3-quantized attention, from first principles
using the paper-best config.

Per-decode-step model (one new query token attends to S cached tokens), summed
over all layers, at context length S. Three KV treatments:
  fp16       : full-precision KV, no residual
  base_int3  : uniform INT3 KV + INT8 group scales (the naive baseline)
  carekv     : base_int3 + sparse 4-bit residual correction (paper-best)

CARE-KV's extra work (vs base_int3):
  (a) router scoring   — score every stored residual candidate with a
                         sketch_dim dot product  -> O(S) FLOPs, tiny constant
  (b) residual read BW — read the top-(RK,RV) selected slots               -> O(1) in S
  (c) scoring read BW  — read K sketches + V error scalars for candidates  -> O(S), tiny
  (d) correction FLOPs — dequant + attn-weighted apply of selected slots   -> O(1) in S

Key results printed: extra FLOPs %, extra read-bandwidth %, and — because decode
is memory-bandwidth bound — the net read BW vs fp16 (CARE-KV is still a large
NET saving over fp16 despite the residual overhead).

Outputs:
  results/overhead_analysis/overhead_flops_bw.csv
  results/overhead_analysis/OVERHEAD_ANALYSIS.md
  results/overhead_analysis/fig_overhead_vs_context.png
"""
from __future__ import annotations
import argparse, csv, math, os

# ── paper-best CARE-KV config (CLAUDE.md §2) ──
CFG = dict(
    page_size=16, group_size=32, k_channel_group=32, v_token_block=4,
    base_bits=3, residual_bits=4, sketch_dim=32,
    sk=2, sv=4, rk=2, rv=2,              # store / read budgets (per page / per step)
    scale_bytes_residual=2,             # fp16 scale per residual slot
)

MODELS = {
    # name: (num_layers, num_q_heads, num_kv_heads, head_dim)
    "Mistral-7B-v0.3": (32, 32, 8, 128),      # GQA (long-context PG-19 model)
    "DeepSeek-7B-base": (30, 32, 32, 128),    # MHA (downstream model)
}

CONTEXTS = [512, 1024, 2048, 4096, 8192]


def per_layer_costs(S, H, Hkv, D, c):
    """Return dict of per-layer, per-decode-step FLOPs and bytes for each mode."""
    P = c["page_size"]; g = c["group_size"]
    Ckg = c["k_channel_group"]; Vtb = c["v_token_block"]
    b = c["base_bits"]; sd = c["sketch_dim"]
    SK, SV, RK, RV = c["sk"], c["sv"], c["rk"], c["rv"]
    n_pages = math.ceil(S / P)
    kcg_per_head = D // Ckg               # K channel-groups per head (=4)
    # slot sizes in bytes (4-bit packed + fp16 scale)
    k_slot_B = P * Ckg * c["residual_bits"] / 8 + c["scale_bytes_residual"]      # 16*32*0.5+2
    v_slot_B = Vtb * D * c["residual_bits"] / 8 + c["scale_bytes_residual"]      # 4*128*0.5+2

    # ── (1) KV-cache read bandwidth per decode step (bytes) ──
    # fp16: read every K and V element (2 bytes each)
    bw_fp16 = 2 * S * Hkv * D * 2
    # base int3: packed codes (b/8 byte each) + int8 group scales
    bw_base = 2 * S * Hkv * D * (b / 8) + 2 * S * Hkv * (D / g) * 1
    # carekv extra vs base:
    #   scoring reads: K sketches (sd elems*2B) for every stored K candidate +
    #                  V error scalar (2B) for every stored V candidate
    score_read = Hkv * n_pages * (SK * sd * 2 + SV * 2)
    #   residual reads: only the selected top-(RK,RV) slots per kv_head (O(1) in S)
    resid_read = Hkv * (RK * k_slot_B + RV * v_slot_B)
    bw_carekv = bw_base + score_read + resid_read

    # ── (2) compute FLOPs per decode step ──
    # attention (shared by all modes): QK^T + softmax + A·V
    flop_attn = 4 * H * S * D + 3 * H * S          # 2*(H*S*D) MAC *2 + softmax ~3*H*S
    # dequant of read KV (shared by base_int3 and carekv): ~1 mul + 1 add / elem
    flop_dequant = 2 * (2 * S * Hkv * D)
    # carekv router scoring: q·r sketch dot (sd MAC) per stored K candidate, per
    # query head in the kv group; V scoring is O(1) per candidate (attn_mass*v_err)
    flop_score = H * n_pages * (SK * sd * 2) + H * n_pages * (SV * 3)
    # carekv correction: dequant + attn-weighted apply of the RK,RV selected slots
    flop_corr = H * (RK * (P * Ckg) + RV * (Vtb * D)) * 2

    return dict(
        bw_fp16=bw_fp16, bw_base=bw_base, bw_carekv=bw_carekv,
        score_read=score_read, resid_read=resid_read,
        flop_attn=flop_attn, flop_dequant=flop_dequant,
        flop_base=flop_attn + flop_dequant,
        flop_carekv=flop_attn + flop_dequant + flop_score + flop_corr,
        flop_score=flop_score, flop_corr=flop_corr,
        flop_fp16=flop_attn,
    )


def kv_footprint_bytes(S, Hkv, D, L, c):
    """Total KV-cache footprint (all layers) for each mode, in bytes."""
    P = c["page_size"]; g = c["group_size"]; b = c["base_bits"]
    Ckg = c["k_channel_group"]; Vtb = c["v_token_block"]
    SK, SV = c["sk"], c["sv"]
    n_pages = math.ceil(S / P)
    k_slot_B = P * Ckg * c["residual_bits"] / 8 + c["scale_bytes_residual"]
    v_slot_B = Vtb * D * c["residual_bits"] / 8 + c["scale_bytes_residual"]
    fp16 = L * 2 * S * Hkv * D * 2
    base = L * (2 * S * Hkv * D * (b / 8) + 2 * S * Hkv * (D / g))
    resid = L * Hkv * n_pages * (SK * k_slot_B + SV * v_slot_B)
    return fp16, base, base + resid


def analyze(model, dims, c):
    L, H, Hkv, D = dims
    rows = []
    for S in CONTEXTS:
        pl = per_layer_costs(S, H, Hkv, D, c)
        # scale per-decode costs to all layers
        flop_fp16 = pl["flop_fp16"] * L
        flop_base = pl["flop_base"] * L
        flop_carekv = pl["flop_carekv"] * L
        bw_fp16 = pl["bw_fp16"] * L
        bw_base = pl["bw_base"] * L
        bw_carekv = pl["bw_carekv"] * L
        fp_mem, base_mem, ck_mem = kv_footprint_bytes(S, Hkv, D, L, c)
        rows.append(dict(
            model=model, context=S,
            # FLOPs
            flop_fp16_M=round(flop_fp16 / 1e6, 3),
            flop_base_M=round(flop_base / 1e6, 3),
            flop_carekv_M=round(flop_carekv / 1e6, 3),
            flop_overhead_vs_base_pct=round(100 * (flop_carekv - flop_base) / flop_base, 3),
            flop_overhead_vs_fp16_pct=round(100 * (flop_carekv - flop_fp16) / flop_fp16, 3),
            # read bandwidth per decode step
            bw_fp16_KB=round(bw_fp16 / 1024, 2),
            bw_base_KB=round(bw_base / 1024, 2),
            bw_carekv_KB=round(bw_carekv / 1024, 2),
            bw_overhead_vs_base_pct=round(100 * (bw_carekv - bw_base) / bw_base, 3),
            bw_carekv_vs_fp16_ratio=round(bw_carekv / bw_fp16, 4),   # <1 = net saving
            # footprint
            kv_fp16_MB=round(fp_mem / 1e6, 2),
            kv_carekv_MB=round(ck_mem / 1e6, 2),
            kv_carekv_vs_fp16_ratio=round(ck_mem / fp_mem, 4),
        ))
    return rows


def make_figure(all_rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[fig] skip ({e})"); return
    models = sorted({r["model"] for r in all_rows})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for model in models:
        rs = [r for r in all_rows if r["model"] == model]
        xs = [r["context"] for r in rs]
        axes[0].plot(xs, [r["flop_overhead_vs_base_pct"] for r in rs], "o-", label=model)
        axes[1].plot(xs, [r["bw_overhead_vs_base_pct"] for r in rs], "o-", label=model)
        axes[2].plot(xs, [100 * r["bw_carekv_vs_fp16_ratio"] for r in rs], "o-", label=model)
    axes[0].set_title("Compute overhead vs INT3 base\n(extra FLOPs %)")
    axes[1].set_title("Read-BW overhead vs INT3 base\n(extra bytes %)")
    axes[2].set_title("CARE-KV read BW as % of fp16\n(<100% = net saving vs fp16)")
    for ax in axes:
        ax.set_xlabel("context length S (tokens)"); ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("%"); axes[1].set_ylabel("%"); axes[2].set_ylabel("% of fp16")
    fig.suptitle("CARE-KV analytical decode-step overhead (paper-best INT3, SK2SV4/RK2RV2)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[fig] wrote {out_png}")


def write_report(all_rows, out_md):
    L = []
    L.append("# CARE-KV — analytical decode-step overhead (FLOPs / memory bandwidth)\n")
    L.append("> Theoretical model (tools/analyze_overhead_flops_bw.py). Separates the "
             "method's **algorithmic** overhead from the current Python-loop **prototype** "
             "wall-clock (results/prefill_decode_perf/), which is an implementation artifact, "
             "not the method cost.\n")
    L.append("\nConfig (paper-best, CLAUDE.md §2): base INT3 + INT8 group scales, 4-bit "
             "residual, page=16, group=32, k_channel_group=32, v_token_block=4, "
             "sketch_dim=32, store SK=2/SV=4, read RK=2/RV=2.\n")
    L.append("\n## Headline\n")
    # summarize at S=4096
    for model in sorted({r["model"] for r in all_rows}):
        r = next(x for x in all_rows if x["model"] == model and x["context"] == 4096)
        L.append(f"- **{model}** @ S=4096: CARE-KV adds **+{r['flop_overhead_vs_base_pct']}% FLOPs** "
                 f"and **+{r['bw_overhead_vs_base_pct']}% read-bandwidth** over plain INT3; its "
                 f"read BW is **{100*r['bw_carekv_vs_fp16_ratio']:.1f}% of fp16** (a net "
                 f"{100-100*r['bw_carekv_vs_fp16_ratio']:.0f}% saving) and its KV footprint is "
                 f"**{100*r['kv_carekv_vs_fp16_ratio']:.1f}% of fp16**.\n")
    L.append("\n## Full table\n")
    cols = ["model", "context", "flop_overhead_vs_base_pct", "flop_overhead_vs_fp16_pct",
            "bw_overhead_vs_base_pct", "bw_carekv_vs_fp16_ratio",
            "bw_fp16_KB", "bw_base_KB", "bw_carekv_KB", "kv_carekv_vs_fp16_ratio"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for r in all_rows:
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    L.append("\n## Interpretation\n")
    L.append("- **Compute overhead is O(S) but tiny-constant.** Router scoring is a "
             "sketch_dim=32 dot product per stored candidate, ~1-2 orders of magnitude "
             "cheaper per element than the O(S·D) attention it rides on; correction touches "
             "only the RK+RV selected slots (O(1) in S).\n")
    L.append("- **Read-bandwidth overhead is dominated by an O(1)-in-S residual read** (the "
             "read budget is a fixed top-(RK,RV) per step) plus an O(S) but very small scoring "
             "read (sketches only). As S grows the overhead % over INT3 base shrinks toward the "
             "residual-read floor.\n")
    L.append("- **Decode is memory-bandwidth bound**, so the relevant number is that CARE-KV "
             "still reads far less than fp16 (INT3 base ≈3/16 of fp16 bytes; residual adds a "
             "small increment) — a large NET bandwidth saving, not a cost.\n")
    L.append("- **The prototype wall-clock (20–100× slower) is Python-loop interpreter "
             "overhead**, not these FLOPs/bytes. A fused kernel (unpack+score+correct) would "
             "realize the small algorithmic overhead above; see "
             "results/prefill_decode_perf/ for the prototype-runtime honesty note.\n")
    open(out_md, "w").write("\n".join(L) + "\n")
    print(f"[md] wrote {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/overhead_analysis")
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)
    all_rows = []
    for model, dims in MODELS.items():
        all_rows += analyze(model, dims, CFG)
    csv_path = os.path.join(A.out_dir, "overhead_flops_bw.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[csv] wrote {csv_path}")
    write_report(all_rows, os.path.join(A.out_dir, "OVERHEAD_ANALYSIS.md"))
    make_figure(all_rows, os.path.join(A.out_dir, "fig_overhead_vs_context.png"))
    # console summary
    for r in all_rows:
        print(f"  {r['model']:18s} S={r['context']:5d}  "
              f"FLOPs +{r['flop_overhead_vs_base_pct']:.2f}% vs base  "
              f"BW +{r['bw_overhead_vs_base_pct']:.2f}% vs base  "
              f"BW {100*r['bw_carekv_vs_fp16_ratio']:.1f}% of fp16")


if __name__ == "__main__":
    main()
