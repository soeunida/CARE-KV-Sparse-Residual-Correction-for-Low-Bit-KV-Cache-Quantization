#!/usr/bin/env bash
# Exp #2: INT2 vs INT3 base × (base_quant, carekv) — does residual correction
# recover more at INT2 (where base quant is worst)? Block-GTQ base ⊕ CARE-KV.
# Δ = carekv - base_quant at each bit width.
set -u
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun:/home/soeun/blockgtq
export TRANSFORMERS_VERBOSITY=error CAREKV_DEBUG_STATS=1
cd /home/soeun/care_kv_clean
GPU=${GPU:-5}; export CUDA_VISIBLE_DEVICES=$GPU
SL=${SL:-64}; N=${N:-4}
OUT=${OUT:-results/int2_carekv/results.csv}
mkdir -p "$(dirname "$OUT")"
echo "[int2] GPU=$GPU SL=$SL N=$N OUT=$OUT  start $(date +%H:%M:%S)"
for bits in 2 3; do
  for mode in base_quant carekv; do
    echo ">>> INT$bits $mode  $(date +%H:%M:%S)"
    python run_blockgtq_carekv.py \
      --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mode "$mode" \
      --seq-len "$SL" --num-samples "$N" --k-avg-bits "$bits" --v-bits "$bits" \
      --seed 0 --append-csv "$OUT" \
      2>&1 | grep -Ev "^  BlockGTQ:" | grep -E "PPL=|Error|Traceback|RuntimeError"
  done
done
echo "[int2] DONE $(date +%H:%M:%S)"
python tools/summarize_bgtq_carekv.py "$OUT" 2>/dev/null || true
