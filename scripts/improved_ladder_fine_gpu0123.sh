#!/usr/bin/env bash
# Follow-on queue for the fine_rb8 ladder, restricted to GPUs 0-3 only.
#
# 13b (GPU 0,1) and deepseek (GPU 2,3) were already launched by
# improved_ladder_fine.sh and are still running as orphans — this script waits on
# their PIDs rather than restarting them. openllama was killed off GPU 4,5 and is
# requeued onto 2,3 as soon as deepseek releases them. Yi-34B runs last on 0-3.
#
# Resumable: run1 skips any model already in the CSV, so re-running this after a
# reboot picks up where it left off (the PID waits simply fall through).
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8
export CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8
export CAREKV_CFG_NAME=fine_rb8

RD=results/blockgtq_carekv
OUT=$RD/improved_ladder_v2.csv
PID_13B=${PID_13B:-41306}
PID_DEEPSEEK=${PID_DEEPSEEK:-41307}

waitpid () {  # poll a non-child PID
  local pid=$1 tag=$2
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
  echo "[g0123] $tag (pid $pid) exited $(date +%m-%d_%H:%M)"
}

run1 () {  # visible_devices model logtag [devmap]
  local dev=$1 model=$2 lg=$3 dm=${4:-}
  if [ -f "$OUT" ] && grep -q "^${model}," "$OUT"; then
    echo "[g0123] === $lg SKIP (already in $OUT) $(date +%m-%d_%H:%M)"; return 0
  fi
  echo "[g0123] >>> $lg START dev=$dev $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$dev CAREKV_DEVICE_MAP=$dm python run_bgtq_adapter.py \
    --model-id "$model" --seq-len 512 --num-samples 32 --append-csv "$OUT" \
    > "$RD/fine_${lg}.log" 2>&1
  echo "[g0123] >>> $lg END rc=$? $(date +%m-%d_%H:%M) :: $(grep 'DONE' "$RD/fine_${lg}.log" | grep -v '^  BlockGTQ:' | tail -1)"
}

echo "[g0123] waiting on in-flight 13b($PID_13B) / deepseek($PID_DEEPSEEK) $(date +%m-%d_%H:%M)"
waitpid "$PID_DEEPSEEK" deepseek
run1 2,3 openlm-research/open_llama_7b_v2 openllama auto
waitpid "$PID_13B" 13b
echo "[g0123] GPUs 0-3 free — starting Yi-34B $(date +%m-%d_%H:%M)"
run1 0,1,2,3 01-ai/Yi-34B Yi-34B auto
echo "[g0123] ALL DONE $(date +%m-%d_%H:%M)"
