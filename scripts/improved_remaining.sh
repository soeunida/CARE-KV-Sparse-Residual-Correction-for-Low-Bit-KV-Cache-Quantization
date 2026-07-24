#!/usr/bin/env bash
# Remaining improved-ladder models with cur_rb8 (8-bit residual at paper budget —
# memory-safe & feasible for MHA/large models; fine_rb8 OOMs / is >28h on MHA).
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
RD=results/blockgtq_carekv; OUT=$RD/improved_ladder.csv
# cur_rb8 config
export CAREKV_CFG_KCG=32 CAREKV_CFG_VTB=4 CAREKV_CFG_KMODE=linear CAREKV_CFG_RB=8
export CAREKV_CFG_SK=2 CAREKV_CFG_SV=4 CAREKV_CFG_RK=2 CAREKV_CFG_RV=2

run1 () {  # visible_devices model logtag [devmap]
  local dev=$1 model=$2 lg=$3 dm=${4:-}
  echo "[rem] >>> $lg START $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$dev CAREKV_DEVICE_MAP=$dm python run_bgtq_adapter.py \
    --model-id "$model" --seq-len 512 --num-samples 32 --append-csv "$OUT" \
    > "$RD/improved_${lg}.log" 2>&1
  echo "[rem] >>> $lg END rc=$? $(date +%m-%d_%H:%M) :: $(grep DONE "$RD/improved_${lg}.log" | grep -v '^  BlockGTQ:' | tail -1)"
}

( run1 6 deepseek-ai/deepseek-llm-7b-base deepseek_rb8 ) &
( run1 5 openlm-research/open_llama_7b_v2 openllama_rb8 ) &
wait
echo "[rem] 7B MHA done — starting Yi-34B (4-GPU) $(date +%m-%d_%H:%M)"
run1 1,4,5,6 01-ai/Yi-34B Yi-34B_rb8 auto
echo "[rem] ALL DONE $(date +%m-%d_%H:%M)"
