#!/usr/bin/env bash
# Exp #4: Read-budget Pareto — quality vs decode read cost. Sweep READ_ABS_K/V
# with STORE fixed at paper (SK2 SV4). READ=(0,0) ≡ base_quant (invariant).
# One CSV per budget so K_reads/V_reads + filename tag the operating point.
set -u
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun:/home/soeun/blockgtq
export TRANSFORMERS_VERBOSITY=error CAREKV_DEBUG_STATS=1
cd /home/soeun/care_kv_clean
GPU=${GPU:-4}; export CUDA_VISIBLE_DEVICES=$GPU
SL=${SL:-64}; N=${N:-4}; BITS=${BITS:-3}
OUTDIR=${OUTDIR:-results/read_budget_pareto}
mkdir -p "$OUTDIR"
M=TinyLlama/TinyLlama-1.1B-Chat-v1.0
echo "[readbudget] GPU=$GPU SL=$SL N=$N BITS=$BITS  start $(date +%H:%M:%S)"

# READ=(0,0): base_quant baseline (residual-off invariant)
echo ">>> READ(0,0)=base_quant  $(date +%H:%M:%S)"
python run_blockgtq_carekv.py --model-id "$M" --mode base_quant \
  --seq-len "$SL" --num-samples "$N" --k-avg-bits "$BITS" --v-bits "$BITS" \
  --seed 0 --append-csv "$OUTDIR/rk0rv0.csv" \
  2>&1 | grep -Ev "^  BlockGTQ:" | grep -E "PPL=|Error|Traceback"

# CARE-KV at increasing read budgets (store fixed at paper SK2 SV4)
for pair in "1 1" "2 2" "4 4"; do
  set -- $pair; rk=$1; rv=$2
  echo ">>> READ($rk,$rv)=carekv  $(date +%H:%M:%S)"
  CAREKV_READ_ABS_K=$rk CAREKV_READ_ABS_V=$rv \
  python run_blockgtq_carekv.py --model-id "$M" --mode carekv \
    --seq-len "$SL" --num-samples "$N" --k-avg-bits "$BITS" --v-bits "$BITS" \
    --seed 0 --append-csv "$OUTDIR/rk${rk}rv${rv}.csv" \
    2>&1 | grep -Ev "^  BlockGTQ:" | grep -E "PPL=|Error|Traceback"
done
echo "[readbudget] DONE $(date +%H:%M:%S)"
