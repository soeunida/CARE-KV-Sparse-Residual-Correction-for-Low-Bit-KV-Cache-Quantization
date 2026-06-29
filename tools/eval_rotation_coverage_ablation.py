"""tools/eval_rotation_coverage_ablation.py

Coverage-routing ablation for the rotation + CARE-KV stack. After rotation the
residual is diffuse, so sparse top-k (K_cap=head_dim/k_channel_group,
V_cap=ceil(page/vtb)) may be wrong. Tests whether broader coverage (finer
granularity → more candidates, stored AND read at full cap) helps the rotated
base more than the uniform base. Memory-matched (rotated vs uniform).

"raise SK/SV cap" == "shrink k_channel_group / v_token_block" (candidate-cap).
Each cell stores AND reads ALL candidates at its granularity (SK=RK=K_cap,
SV=RV=V_cap) = full coverage. `uni_bar` is the paper-best uniform+CARE-KV RV2.

DIAGNOSTIC. Outputs: <out-csv>.
"""
from __future__ import annotations
import argparse, csv, math, os, sys, time

sys.path.insert(0, "/home/soeun")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from CARE_KV.care_kv.baselines import FP16Adapter, BaseQuantAdapter, CAREKVAdapter
from eval_base_quantizer_expansion import run_one

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HEAD_DIM, PAGE = 64, 16
MAXP = 16


def _caps(kcg, vtb):
    return HEAD_DIM // kcg, math.ceil(PAGE / vtb)


def _cov_adapter(base, kmode, kcg, vtb):
    """Full-coverage CARE-KV: store AND read every candidate at this granularity."""
    kcap, vcap = _caps(kcg, vtb)
    a = CAREKVAdapter(mode="fixed", bits=3, base_quantizer=base,
                      k_store_mode=kmode, bits_k=3, bits_v=3,
                      sk=kcap, sv=vcap, rk=kcap, rv=vcap,
                      k_channel_group=kcg, v_token_block=vtb, max_pages=MAXP)
    short = {"rotatekv_style": "rot", "randrot_style": "rand",
             "uniform": "uni"}.get(base, base)
    a.name = f"{short}_g{kcg}v{vtb}_SK{kcap}SV{vcap}_cover"
    return a


def _bar_adapter():
    """The paper-best uniform + CARE-KV bar (fixed, g32v4, RV2)."""
    a = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                      k_store_mode="post_rope", bits_k=3, bits_v=3,
                      sk=2, sv=4, rk=2, rv=2,
                      k_channel_group=32, v_token_block=4, max_pages=MAXP)
    a.name = "uni_bar_g32v4_RV2"
    return a


CELLS = {
    "fp16":        lambda: FP16Adapter(),
    "base_int3":   lambda: BaseQuantAdapter(bits=3),
    "uni_bar":     lambda: _bar_adapter(),
    "rot_g32v4":   lambda: _cov_adapter("rotatekv_style", "pre_rope", 32, 4),
    "rot_g16v2":   lambda: _cov_adapter("rotatekv_style", "pre_rope", 16, 2),
    "uni_g32v4":   lambda: _cov_adapter("uniform", "post_rope", 32, 4),
    "uni_g16v2":   lambda: _cov_adapter("uniform", "post_rope", 16, 2),
}
DEFAULT = list(CELLS.keys())


def main():
    global MODEL_ID, MAXP, HEAD_DIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--head-dim", type=int, default=HEAD_DIM,
                    help="model head_dim (TinyLlama 64, DeepSeek/Mistral 128)")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--cells", default=",".join(DEFAULT))
    args = ap.parse_args()
    MODEL_ID = args.model_id
    HEAD_DIM = args.head_dim
    MAXP = max(MAXP, args.seq_len // PAGE + 4)

    chosen = [c.strip() for c in args.cells.split(",") if c.strip()]
    bad = [c for c in chosen if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cells {bad}; known {list(CELLS)}")

    rows = []
    for c in chosen:
        a = CELLS[c]()
        t0 = time.perf_counter()
        r = run_one(a, MODEL_ID, "wikitext", args.seq_len, args.num_samples)
        print(f"[COVER] {c:12s} {a.name:34s} PPL={r.ppl:9.4f} "
              f"resMB={r.residual_memory_MB:7.3f} "
              f"K_reads={r.k_reads:>8d} V_reads={r.v_reads:>8d} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
        d = r.as_dict(); d["cell"] = c
        rows.append(d)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = []
    for d in rows:
        for k in d:
            if k not in keys: keys.append(k)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for d in rows: w.writerow(d)
    print(f"wrote {len(rows)} rows -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
