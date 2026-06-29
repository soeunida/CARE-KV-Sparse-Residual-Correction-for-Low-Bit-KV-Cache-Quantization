from pathlib import Path
import re
import sys

paths = [Path(p) for p in sys.argv[1:]]
if not paths:
    paths = sorted(Path("results").glob("ppl_*.txt"))

rows = []

def find(pattern, text, default=""):
    m = re.search(pattern, text)
    return m.group(1) if m else default

for p in paths:
    if not p.exists():
        continue
    text = p.read_text(errors="ignore")
    model = find(r"MODEL=(.*)", text)
    mode = find(r"MODE=(.*)", text)
    bits = find(r"BASE_BITS=(.*)", text)
    seq = find(r"SEQ_LEN=(.*)", text)
    toks = find(r"TOKENS=(.*)", text)
    ppl = find(r"PPL=([0-9.]+)", text)
    if ppl:
        rows.append((p.name, model, mode, bits, seq, toks, float(ppl)))

if not rows:
    print("No PPL results found.")
    raise SystemExit(0)

print("| file | mode | bits | seq_len | tokens | PPL |")
print("|---|---|---:|---:|---:|---:|")
for name, model, mode, bits, seq, toks, ppl in rows:
    print(f"| {name} | {mode} | {bits} | {seq} | {toks} | {ppl:.4f} |")
