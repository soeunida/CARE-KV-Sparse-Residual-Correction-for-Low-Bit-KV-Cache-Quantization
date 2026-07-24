#!/usr/bin/env bash
set -u
gpu=$1; model=$2
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh; conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=$gpu
export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8
export CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8
tag=$(basename "$model")
python run_bgtq_adapter.py --model-id "$model" --seq-len 512 --num-samples 32 \
  --append-csv results/blockgtq_carekv/improved_ladder.csv > results/blockgtq_carekv/improved_${tag}.log 2>&1
echo "improved $tag done"
