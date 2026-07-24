#!/usr/bin/env bash
# Finish the ladder, take 2. History on 13b:
#   fine_rb8 + device_map shard -> C-arm HUNG (20h, GPU 0%)
#   cur_rb8  + single GPU       -> C-arm OOM (2 model copies > 48GB)
# This run: cur_rb8 (light linear correction) + 2-GPU shard on 0,1 = memory
# headroom to avoid OOM, light correction to avoid the fine_rb8 hang.
# GPU 1 also hosts soeun's SFPA run (~16GB); 32GB free there is plenty.
# Plan (GPUs 0-3 only):
#   13b       cur_rb8  shard 0,1   (starts now)
#   deepseek  fine_rb8 shard 2,3   (already running, pid 41307)
#   openllama fine_rb8 shard 2,3   (after deepseek frees them)
#   Yi-34B    cur_rb8  shard 0-3   (after 13b + openllama)
# Resumable: _run skips any model already in the CSV.
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RD=results/blockgtq_carekv
OUT=$RD/improved_ladder_v2.csv
PID_DEEPSEEK=${PID_DEEPSEEK:-41307}

_run () {  # cfgname devices devmap logtag model  (config env set by caller)
  local cfgname=$1 dev=$2 dm=$3 lg=$4 model=$5
  if [ -f "$OUT" ] && grep -q "^${model}," "$OUT"; then
    echo "[v2] === $lg SKIP (already in $OUT) $(date +%m-%d_%H:%M)"; return 0
  fi
  echo "[v2] >>> $lg ($cfgname) START dev=$dev dm=$dm $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$dev CAREKV_DEVICE_MAP=$dm CAREKV_CFG_NAME=$cfgname \
    python run_bgtq_adapter.py --model-id "$model" --seq-len 512 --num-samples 32 \
    --append-csv "$OUT" > "$RD/v2_${lg}.log" 2>&1
  echo "[v2] >>> $lg END rc=$? $(date +%m-%d_%H:%M) :: $(grep 'DONE' "$RD/v2_${lg}.log" | grep -v '^  BlockGTQ:' | tail -1)"
}
run_cur () {  ( export CAREKV_CFG_KCG=32 CAREKV_CFG_VTB=4 CAREKV_CFG_KMODE=linear CAREKV_CFG_RB=8 \
                       CAREKV_CFG_SK=2 CAREKV_CFG_SV=4 CAREKV_CFG_RK=2 CAREKV_CFG_RV=2
                _run cur_rb8 "$1" "$2" "$3" "$4" ); }
run_fine () { ( export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8 \
                       CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8
                _run fine_rb8 "$1" "$2" "$3" "$4" ); }
waitpid () { local pid=$1 tag=$2; while kill -0 "$pid" 2>/dev/null; do sleep 60; done
             echo "[v2] $tag (pid $pid) exited $(date +%m-%d_%H:%M)"; }

# 1) 13b cur_rb8 sharded on GPU 0,1 (device_map=auto) — 2 GPUs avoid the OOM.
run_cur 0,1 auto 13b NousResearch/Llama-2-13b-hf &
PID_13B=$!

# 2) openllama fine_rb8 on 2,3 after deepseek frees them.
echo "[v2] waiting on deepseek (pid $PID_DEEPSEEK) to free GPU 2,3 $(date +%m-%d_%H:%M)"
waitpid "$PID_DEEPSEEK" deepseek
run_fine 2,3 auto openllama openlm-research/open_llama_7b_v2

# 3) Yi-34B cur_rb8 on 0-3 once 13b and openllama are done.
wait "$PID_13B"
echo "[v2] 13b + openllama done — starting Yi-34B (cur_rb8, 4-GPU) $(date +%m-%d_%H:%M)"
run_cur 0,1,2,3 auto Yi-34B 01-ai/Yi-34B
echo "[v2] ALL DONE $(date +%m-%d_%H:%M)"
