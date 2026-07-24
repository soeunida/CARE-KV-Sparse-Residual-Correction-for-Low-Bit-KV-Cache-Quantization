#!/usr/bin/env bash
# openllama fine_rb8 SHARDED, run NOW in parallel with Yi-34B fine (user OK'd
# using free GPUs). Requested 4,5 but GPU 5 is now taken by another user (miyeon,
# 91%), so shard on the two actually-free GPUs 0,4 — same intent (parallel, 2-GPU
# to dodge the single-GPU fine_rb8 hang, no collision with others' jobs).
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
if grep -q "^openlm-research/open_llama_7b_v2," "$OUT" 2>/dev/null; then echo "[olp] SKIP already in CSV"; exit 0; fi
echo "[olp] START openllama fine_rb8 SHARDED GPU 0,4 $(date +%m-%d_%H:%M)"
CUDA_VISIBLE_DEVICES=0,4 CAREKV_DEVICE_MAP=auto python run_bgtq_adapter.py \
  --model-id openlm-research/open_llama_7b_v2 --seq-len 512 --num-samples 32 \
  --append-csv "$OUT" > "$RD/ol_parallel.log" 2>&1
echo "[olp] END rc=$? $(date +%m-%d_%H:%M) :: $(grep DONE $RD/ol_parallel.log | tail -1)"
