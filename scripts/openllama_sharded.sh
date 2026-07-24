#!/usr/bin/env bash
# Re-run openllama fine_rb8 SHARDED, after the single-GPU C-arm hung (56min, GPU
# 1%, wchan=0 spin — same as 13b; single-GPU fine_rb8 hangs, sharded completes).
# deepseek is the one confirmed fine_rb8 completion and it ran device_map=auto on
# 2 GPUs. Wait for Yi-34B fine (pid $PID_YI) to free GPUs 1,2,3, then shard on 1,2.
# Resumable: skips if openllama already in the CSV.
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8
export CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8
export CAREKV_CFG_NAME=fine_rb8
RD=results/blockgtq_carekv; OUT=$RD/improved_ladder_v2.csv
PID_YI=${PID_YI:-1315931}

if grep -q "^openlm-research/open_llama_7b_v2," "$OUT" 2>/dev/null; then echo "[ol] SKIP already in CSV"; exit 0; fi
echo "[ol] waiting on Yi-34B fine (pid $PID_YI) to free GPUs $(date +%m-%d_%H:%M)"
while kill -0 "$PID_YI" 2>/dev/null; do sleep 60; done
echo "[ol] Yi done — starting openllama fine_rb8 SHARDED on GPU 1,2 $(date +%m-%d_%H:%M)"
CUDA_VISIBLE_DEVICES=1,2 CAREKV_DEVICE_MAP=auto python run_bgtq_adapter.py \
  --model-id openlm-research/open_llama_7b_v2 --seq-len 512 --num-samples 32 \
  --append-csv "$OUT" > "$RD/ol_sharded.log" 2>&1
echo "[ol] END rc=$? $(date +%m-%d_%H:%M) :: $(grep DONE $RD/ol_sharded.log | tail -1)"
