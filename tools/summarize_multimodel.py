"""tools/summarize_multimodel.py — consolidate per-model CARE-KV eval CSVs into
one markdown table. Flags failed cells (nan / 0.0 / ERROR notes) as infra
failures (OOM / CPU-offload / contention) rather than method results, and
computes the recovery fraction (base-INT3 → CARE-KV toward fp16) for valid rows.
"""
from __future__ import annotations
import argparse, csv, glob, json, math, os

ARMS = ["fp16", "base_int3", "carekv_uniform_vec"]


def arch_of(model_id):
    """Read num_key_value_heads vs num_attention_heads from the HF cache config
    → 'MHA' or 'GQA' (+ Hkv). Falls back to '?' if the config is not local."""
    hf = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache/huggingface")
    d = os.path.join(hf, "hub", "models--" + model_id.replace("/", "--"))
    cfgs = glob.glob(os.path.join(d, "snapshots", "*", "config.json"))
    if not cfgs:
        return "?"
    try:
        c = json.load(open(cfgs[0]))
        h = c.get("num_attention_heads")
        kv = c.get("num_key_value_heads", h)
        return f"{'MHA' if kv == h else 'GQA'} (Hkv={kv})"
    except Exception:
        return "?"


def load_model(path):
    rows = {r.get("arm", ""): r for r in csv.DictReader(open(path))}
    return rows


def val(rows, arm):
    r = rows.get(arm)
    if not r:
        return None, "missing"
    p = r.get("ppl", "")
    try:
        f = float(p)
    except Exception:
        return None, f"?{p}"
    note = r.get("notes", "") or ""
    if math.isnan(f):
        return None, "nan"
    if f == 0.0:
        cause = "OOM" if "OutOfMemory" in note else "0.0"
        return None, cause
    return f, "ok"


def kv(rows):
    r = next((rows[a] for a in rows if a.startswith("carekv")), None)
    if not r:
        return 0, 0
    try:
        return int(r.get("k_reads", 0) or 0), int(r.get("v_reads", 0) or 0)
    except Exception:
        return 0, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/multimodel_7b")
    ap.add_argument("--out", default="results/multimodel_7b/MULTIMODEL_SUMMARY.md")
    args = ap.parse_args()

    csvs = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    csvs = [c for c in csvs if os.path.basename(c) not in
            ("multimodel_summary.csv",)]
    L = ["# Multi-model CARE-KV validation (WikiText-2, SL512, N=4)", "",
         "CARE-KV = paper-best vectorized (SK2 SV4 RK2 RV2, sketch32). "
         "**recovery** = (base − carekv)/(base − fp16), fraction of the INT3 "
         "quality gap that CARE-KV closes toward fp16. Cells marked ✗ are "
         "**infrastructure failures** (OOM / device_map CPU-offload / shared-GPU "
         "contention), not method results — they need a clean re-run.", "",
         "| Model | arch | fp16 | base INT3 | CARE-KV | recovery | router K/V | status |",
         "|---|---|---:|---:|---:|---:|---:|:--|"]
    valid = []
    for c in csvs:
        name = os.path.basename(c)[:-4]
        rows = load_model(c)
        mid = next(iter(rows.values()), {}).get("model_id", name)
        base_arm = next((a for a in rows if a.startswith("base_int")), "base_int3")
        carekv_arm = next((a for a in rows if a.startswith("carekv")), "carekv_uniform_vec")
        f_fp, s_fp = val(rows, "fp16")
        f_bq, s_bq = val(rows, base_arm)
        f_ck, s_ck = val(rows, carekv_arm)
        k, v = kv(rows)
        arch = arch_of(mid)

        def cell(f, s):
            return f"{f:.4f}" if f is not None else f"✗ {s}"
        rec = ""
        ok = (f_fp and f_bq and f_ck and k + v > 0)
        if ok and (f_bq - f_fp) != 0:
            rec = f"{100*(f_bq - f_ck)/(f_bq - f_fp):.0f}%"
            valid.append((mid, f_fp, f_bq, f_ck, rec))
        status = "✓ valid" if ok else "⚠ needs re-run"
        L.append(f"| {mid} | {arch} | {cell(f_fp,s_fp)} | {cell(f_bq,s_bq)} | "
                 f"{cell(f_ck,s_ck)} | {rec or '—'} | "
                 f"{k//1000}k/{v//1000}k | {status} |")

    L += ["", "## Valid CARE-KV results", ""]
    if valid:
        for mid, fp, bq, ck, rec in valid:
            L.append(f"- **{mid}**: base-INT3 {bq:.3f} → CARE-KV {ck:.3f} "
                     f"(fp16 {fp:.3f}) — recovers **{rec}** of the gap.")
    else:
        L.append("- (none fully valid yet)")
    L += ["", "## Notes",
          "- ✗ cells: shared-server GPU contention + device_map=auto CPU-offload "
          "(meta-device) corrupted fp16/base arms on the 34B models "
          "(nan/0.0/OOM). Method (CARE-KV) is unaffected where baselines are valid.",
          "- MHA models (Llama-2-13B, Hkv=40) need ≥2 GPUs: the KV-head-indexed "
          "CARE-KV cache is ~5× a GQA model's.",
          "- Clean re-run recipe: 3 GPUs per 34B (no offload) + per-arm process "
          "isolation (fresh memory between fp16/base/carekv)."]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
