#!/usr/bin/env bash
# Finish the ladder after the fine_rb8 13b C-arm hung (20h, GPU 0% — likely a
# device_map-sharding CPU fallback). Per user decision 07-21:
#   13b, Yi-34B  -> cur_rb8   (fast, memory-safe)
#   deepseek, openllama -> fine_rb8   (deepseek already running healthy)
# 13b cur_rb8 runs on a SINGLE GPU (no sharding) to dodge the suspected hang.
# All rows carry a config column (CAREKV_CFG_NAME), so the mix is labeled.
# Resumable: run_* skips any model already in the CSV.
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
    echo "[fin] === $lg SKIP (already in $OUT) $(date +%m-%d_%H:%M)"; return 0
  fi
  echo "[fin] >>> $lg ($cfgname) START dev=$dev dm=$dm $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$dev CAREKV_DEVICE_MAP=$dm CAREKV_CFG_NAME=$cfgname \
    python run_bgtq_adapter.py --model-id "$model" --seq-len 512 --num-samples 32 \
    --append-csv "$OUT" > "$RD/fin_${lg}.log" 2>&1
  echo "[fin] >>> $lg END rc=$? $(date +%m-%d_%H:%M) :: $(grep 'DONE' "$RD/fin_${lg}.log" | grep -v '^  BlockGTQ:' | tail -1)"
}

run_cur () {  # devices devmap logtag model
  ( export CAREKV_CFG_KCG=32 CAREKV_CFG_VTB=4 CAREKV_CFG_KMODE=linear CAREKV_CFG_RB=8 \
           CAREKV_CFG_SK=2 CAREKV_CFG_SV=4 CAREKV_CFG_RK=2 CAREKV_CFG_RV=2
    _run cur_rb8 "$1" "$2" "$3" "$4" )
}
run_fine () {  # devices devmap logtag model
  ( export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8 \
           CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8
    _run fine_rb8 "$1" "$2" "$3" "$4" )
}
waitpid () { local pid=$1 tag=$2; while kill -0 "$pid" 2>/dev/null; do sleep 60; done
             echo "[fin] $tag (pid $pid) exited $(date +%m-%d_%H:%M)"; }

# 1) 13b cur_rb8 on single GPU 0 (no device_map) — starts immediately, in bg.
run_cur 0 "" 13b NousResearch/Llama-2-13b-hf &
PID_13B=$!

# 2) openllama fine_rb8 on GPU 2,3 once deepseek frees them.
echo "[fin] waiting on deepseek (pid $PID_DEEPSEEK) to free GPU 2,3 $(date +%m-%d_%H:%M)"
waitpid "$PID_DEEPSEEK" deepseek
run_fine 2,3 auto openllama openlm-research/open_llama_7b_v2

# 3) Yi-34B cur_rb8 on all of 0-3 once 13b and openllama are done.
wait "$PID_13B"
echo "[fin] 13b + openllama done — starting Yi-34B (cur_rb8, 4-GPU) $(date +%m-%d_%H:%M)"
run_cur 0,1,2,3 auto Yi-34B 01-ai/Yi-34B
echo "[fin] ALL DONE $(date +%m-%d_%H:%M)"
