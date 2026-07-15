"""Summarize Block-GTQ ⊕ CARE-KV WikiText-2 PPL runs into a B / C / Δ table.

Reads the CSV written by run_blockgtq_carekv.py and, for each
(model, seq_len), reports:
    B = base_quant_blockgtq PPL           (Block-GTQ K3V3 baseline)
    C = carekv_blockgtq  PPL              (+ CARE-KV residual)
    Δ = C - B                             (negative = residual helps)
    standalone_blockgtq PPL               (independent cross-check of B)
    K_reads / V_reads for C               (router must have fired: > 0)

Usage:
    python tools/summarize_bgtq_carekv.py results/blockgtq_carekv/results.csv
"""
import csv, sys
from collections import defaultdict


def short(m):
    return m.split("/")[-1]


def main(path):
    rows = list(csv.DictReader(open(path)))
    # key: (model, seq_len) -> mode -> row
    by = defaultdict(dict)
    for r in rows:
        by[(r["model"], int(r["seq_len"]))][r["mode"]] = r

    hdr = (f"{'model':<26} {'SL':>5} {'N':>4} | {'B(Block-GTQ)':>13} "
           f"{'C(+CARE-KV)':>13} {'Δ=C-B':>9} {'Δ%':>7} | "
           f"{'standalone':>11} {'K_rd':>6} {'V_rd':>6}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for (model, sl) in sorted(by, key=lambda k: (short(k[0]), k[1])):
        d = by[(model, sl)]
        b = d.get("base_quant_blockgtq")
        c = d.get("carekv_blockgtq")
        s = d.get("standalone_blockgtq")
        B = float(b["ppl"]) if b else None
        C = float(c["ppl"]) if c else None
        S = float(s["ppl"]) if s else None
        N = (b or c or s)["num_samples"]
        delta = (C - B) if (B is not None and C is not None) else None
        dpct = (100.0 * delta / B) if (delta is not None and B) else None
        Kr = c["K_reads"] if c else "-"
        Vr = c["V_reads"] if c else "-"
        print(f"{short(model):<26} {sl:>5} {N:>4} | "
              f"{('%.4f'%B) if B is not None else '-':>13} "
              f"{('%.4f'%C) if C is not None else '-':>13} "
              f"{('%+.4f'%delta) if delta is not None else '-':>9} "
              f"{('%+.2f%%'%dpct) if dpct is not None else '-':>7} | "
              f"{('%.4f'%S) if S is not None else '-':>11} {Kr:>6} {Vr:>6}")
        out.append(dict(model=short(model), seq_len=sl, num_samples=N,
                        B=B, C=C, delta=delta, delta_pct=dpct,
                        standalone=S, K_reads=Kr, V_reads=Vr))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "results/blockgtq_carekv/results.csv")
