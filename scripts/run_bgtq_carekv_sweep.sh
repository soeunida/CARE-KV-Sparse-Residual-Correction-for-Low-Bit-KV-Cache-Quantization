#!/usr/bin/env bash
# Block-GTQ ⊕ CARE-KV WikiText-2 PPL sweep: B (base_quant), C (carekv), Δ=C-B.
# Fixed: K3V3, fixed seed, no fp16 recent-key buffer, WT2-train calib / WT2-test eval.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_bgtq_carekv_sweep.sh
# Env overrides: OUT, SEED, GPU, MODELS, SEQLENS, NTINY, NBIG
set -u
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun:/home/soeun/blockgtq
export TRANSFORMERS_VERBOSITY=error
cd /home/soeun/care_kv_clean

OUT=${OUT:-results/blockgtq_carekv/results.csv}
SEED=${SEED:-0}
SEQLENS=${SEQLENS:-"512 1024"}
# model_id:N  (N = num windows; smaller for big models to bound runtime)
MODELS=${MODELS:-"TinyLlama/TinyLlama-1.1B-Chat-v1.0:32 mistralai/Mistral-7B-v0.3:16"}

echo "[sweep] OUT=$OUT SEED=$SEED SEQLENS=$SEQLENS"
echo "[sweep] MODELS=$MODELS"

run () {  # model  N  seqlen  mode
  local model=$1 n=$2 sl=$3 mode=$4
  echo ">>> $(basename "$model") SL=$sl N=$n mode=$mode  $(date +%H:%M:%S)"
  python run_blockgtq_carekv.py --model-id "$model" --mode "$mode" \
    --seq-len "$sl" --num-samples "$n" --seed "$SEED" --append-csv "$OUT" \
    2>&1 | grep -Ev "^  BlockGTQ:" | grep -E "PPL=|Error|Traceback|RuntimeError|assert"
}

for entry in $MODELS; do
  model=${entry%:*}; n=${entry##*:}
  for sl in $SEQLENS; do
    run "$model" "$n" "$sl" standalone
    run "$model" "$n" "$sl" base_quant
    run "$model" "$n" "$sl" carekv
  done
done

echo "[sweep] DONE  $(date +%H:%M:%S)"
python tools/summarize_bgtq_carekv.py "$OUT"
