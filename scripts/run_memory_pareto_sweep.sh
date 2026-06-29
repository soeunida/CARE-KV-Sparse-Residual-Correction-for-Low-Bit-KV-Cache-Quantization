#!/usr/bin/env bash
#
# scripts/run_memory_pareto_sweep.sh
# -----------------------------------
# Residual-granularity Pareto sweep for INT3 carekv_stored.
#
# Sweeps:
#   page_size, v_token_block, k_channel_group, sketch_dim
#
# Writes CSV + markdown table to:
#   results/memory_pareto_sweep_<ts>.csv
#   results/memory_pareto_sweep_<ts>.md
#
# Env:
#   SEQ_LEN — sequence length (default 128)
#   MODEL_ID (default TinyLlama)
#
set -euo pipefail

cd "$(dirname "$0")/.."

source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

TS=$(date +%Y%m%d_%H%M%S)
SEQ_LEN="${SEQ_LEN:-128}"
OUT_DIR="results"
CSV_PATH="$OUT_DIR/memory_pareto_sweep_${TS}.csv"
MD_PATH="$OUT_DIR/memory_pareto_sweep_${TS}.md"

mkdir -p "$OUT_DIR"

echo "=== memory-pareto sweep (SEQ_LEN=$SEQ_LEN, INT3 carekv_stored, V-only) ==="
python tools/paper_eval.py memory-pareto \
  --seq-len "$SEQ_LEN" \
  --out-csv "$CSV_PATH"

# Generate a markdown summary from the CSV.
python - <<PY
import csv, sys
rows = list(csv.DictReader(open("$CSV_PATH")))
if not rows:
    open("$MD_PATH","w").write("# Memory Pareto sweep (no rows)\n")
    sys.exit(0)
out = []
out.append("# Memory Pareto sweep — INT3 carekv_stored, V-only")
out.append("")
out.append(f"SEQ_LEN = $SEQ_LEN, store_budget = first row's value, read_budget = first row's value")
out.append("")
keep_cols = ["label","page_size","v_token_block","k_channel_group","sketch_dim",
             "store_budget","read_budget","ppl","total_MB","vs_fp16",
             "base_MB","scale_MB","residual_MB","metadata_MB","sketch_MB",
             "v_slots_read","k_slots_read","actual_read_ratio","seconds"]
fmt = {"store_budget":"{:.2f}","read_budget":"{:.2f}","ppl":"{:.4f}",
       "total_MB":"{:.2f}","vs_fp16":"{:.3f}","base_MB":"{:.2f}",
       "scale_MB":"{:.2f}","residual_MB":"{:.2f}","metadata_MB":"{:.2f}",
       "sketch_MB":"{:.2f}","actual_read_ratio":"{:.2f}","seconds":"{:.1f}"}
out.append("| " + " | ".join(keep_cols) + " |")
out.append("|" + "|".join("---" for _ in keep_cols) + "|")
for r in rows:
    cells = []
    for c in keep_cols:
        v = r.get(c, "")
        if c in fmt and v not in ("", None):
            try: cells.append(fmt[c].format(float(v)))
            except (ValueError, TypeError): cells.append(str(v))
        else:
            cells.append(str(v))
    out.append("| " + " | ".join(cells) + " |")

# Pareto highlights
out.append("")
out.append("## Pareto highlights")
valid = [r for r in rows if "ppl" in r and r.get("ppl","") not in ("","nan")]
def fnum(r,k,d=float("inf")):
    try: return float(r.get(k,d))
    except: return d

if valid:
    by_mem = sorted(valid, key=lambda r: fnum(r,"total_MB"))
    by_ppl = sorted(valid, key=lambda r: fnum(r,"ppl"))
    by_ratio = sorted(valid, key=lambda r: fnum(r,"total_MB") * fnum(r,"ppl"))
    by_speed = sorted(valid, key=lambda r: fnum(r,"seconds"))
    out.append(f"- **Lowest memory**: {by_mem[0]['label']} → {by_mem[0]['total_MB']} MB at PPL {by_mem[0]['ppl']}")
    out.append(f"- **Best PPL**: {by_ppl[0]['label']} → PPL {by_ppl[0]['ppl']} at {by_ppl[0]['total_MB']} MB")
    out.append(f"- **Best memory × PPL**: {by_ratio[0]['label']} → {by_ratio[0]['total_MB']} MB × PPL {by_ratio[0]['ppl']}")
    out.append(f"- **Fastest forward**: {by_speed[0]['label']} → {by_speed[0]['seconds']} s")

open("$MD_PATH","w").write("\n".join(out)+"\n")
print(f"wrote {len(rows)} rows to $MD_PATH")
PY

echo
echo "============================================================"
echo "  CSV : $CSV_PATH"
echo "  MD  : $MD_PATH"
echo "============================================================"
