#!/usr/bin/env bash
# DIAGNOSTIC PROBE (throwaway numbers): does 13b's C-arm complete under
# correction_impl=cached, where vectorized CPU-spins/hangs on all configs?
# Small N=4 + reduced calib (512 tok) to reach the C-arm fast (~20min) — the
# PPL is not paper-valid, we only care whether [C carekv] prints (=cached works)
# vs GPU~0% userspace spin (=cached also hangs). GPU 1 (free, dedicated).
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cur_rb8 residual config + cached correction + fast probe calib
export CAREKV_CFG_KCG=32 CAREKV_CFG_VTB=4 CAREKV_CFG_KMODE=linear CAREKV_CFG_RB=8
export CAREKV_CFG_SK=2 CAREKV_CFG_SV=4 CAREKV_CFG_RK=2 CAREKV_CFG_RV=2
export CAREKV_CFG_CORR_IMPL=cached CAREKV_CFG_NCALIB=512 CAREKV_CFG_NAME=probe_cur_cached
RD=results/blockgtq_carekv
echo "[probe] START $(date +%m-%d_%H:%M) — 13b cur_rb8 cached N=4 ncalib=512 GPU1"
CUDA_VISIBLE_DEVICES=1 python run_bgtq_adapter.py \
  --model-id NousResearch/Llama-2-13b-hf --seq-len 512 --num-samples 4 \
  --append-csv "$RD/probe_13b_cached.csv" > "$RD/probe_13b_cached.log" 2>&1
echo "[probe] END rc=$? $(date +%m-%d_%H:%M) :: $(grep -E 'DONE|ERROR' $RD/probe_13b_cached.log | tail -1)"
