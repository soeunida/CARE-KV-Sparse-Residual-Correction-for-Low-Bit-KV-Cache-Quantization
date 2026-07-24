#!/usr/bin/env bash
# Ladder take 4 — run the two REMAINING feasible models in parallel within GPUs
# 0-3, after deepseek completed (first v2 row). 13b is EXCLUDED (pending user
# decision: both vectorized[hang] and cached[crawl, 15h for N=4] are impractical
# for its full-MHA 40-KV-head C-arm = 1600 layer×head units).
#   openllama fine_rb8  GPU 0 single  (7B, ~27h like deepseek; single GPU = no OOM/no shard-hang)
#   Yi-34B    cur_rb8   GPU 1,2,3     (GQA nkv=8 → 480 units, lighter than deepseek; needs 3 GPUs for 34B)
# Both use vectorized (default). Resumable: skips models already in the CSV.
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RD=results/blockgtq_carekv
OUT=$RD/improved_ladder_v2.csv

_run () {  # cfgname devices devmap logtag model
  local cfgname=$1 dev=$2 dm=$3 lg=$4 model=$5
  if [ -f "$OUT" ] && grep -q "^${model}," "$OUT"; then
    echo "[v4] === $lg SKIP (already in $OUT) $(date +%m-%d_%H:%M)"; return 0
  fi
  echo "[v4] >>> $lg ($cfgname) START dev=$dev dm=$dm $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$dev CAREKV_DEVICE_MAP=$dm CAREKV_CFG_NAME=$cfgname \
    python run_bgtq_adapter.py --model-id "$model" --seq-len 512 --num-samples 32 \
    --append-csv "$OUT" > "$RD/v4_${lg}.log" 2>&1
  echo "[v4] >>> $lg END rc=$? $(date +%m-%d_%H:%M) :: $(grep 'DONE' "$RD/v4_${lg}.log" | grep -v '^  BlockGTQ:' | tail -1)"
}
run_cur () {  ( export CAREKV_CFG_KCG=32 CAREKV_CFG_VTB=4 CAREKV_CFG_KMODE=linear CAREKV_CFG_RB=8 \
                       CAREKV_CFG_SK=2 CAREKV_CFG_SV=4 CAREKV_CFG_RK=2 CAREKV_CFG_RV=2
                _run cur_rb8 "$1" "$2" "$3" "$4" ); }
run_fine () { ( export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8 \
                       CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8
                _run fine_rb8 "$1" "$2" "$3" "$4" ); }

run_fine 0   ""   openllama openlm-research/open_llama_7b_v2 &
run_cur  1,2,3 auto Yi-34B  01-ai/Yi-34B &
wait
echo "[v4] ALL DONE $(date +%m-%d_%H:%M)"
