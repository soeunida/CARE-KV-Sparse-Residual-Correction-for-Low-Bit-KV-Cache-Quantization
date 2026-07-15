#!/usr/bin/env bash
# Run all 3 modes (standalone, base_quant, carekv) for ONE (model, SL, N) cell
# on a given GPU, appending to the shared results CSV.
# Usage: cell.sh <gpu> <model_id> <seq_len> <N> <out_csv> [seed]
set -u
gpu=$1; model=$2; sl=$3; n=$4; out=$5; seed=${6:-0}
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun:/home/soeun/blockgtq
export TRANSFORMERS_VERBOSITY=error
export CUDA_VISIBLE_DEVICES=$gpu
cd /home/soeun/care_kv_clean
tag="$(basename "$model")|SL$sl|N$n|gpu$gpu"
for mode in standalone base_quant carekv; do
  echo ">>> [$tag] $mode START $(date +%H:%M:%S)"
  python run_blockgtq_carekv.py --model-id "$model" --mode "$mode" \
    --seq-len "$sl" --num-samples "$n" --seed "$seed" --append-csv "$out" \
    2>&1 | grep -Ev "^  BlockGTQ:" | grep -E "PPL=|Error|Traceback|RuntimeError|assert|Killed"
  echo ">>> [$tag] $mode END   $(date +%H:%M:%S)"
done
echo ">>> [$tag] CELL DONE $(date +%H:%M:%S)"
