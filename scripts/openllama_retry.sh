#!/usr/bin/env bash
# openllama fine_rb8 retry SHARDED on the freshly-freed clean GPUs 1,2 (Yi killed).
# Prior attempt OOMed because the user's RAG/SFPA jobs crowded GPU 0. GPU 1,2 are
# clean now; GPU 0=RAG, 4=SFPA, 5,6=miyeon — avoid all of those.
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh; conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8
export CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8 CAREKV_CFG_NAME=fine_rb8
RD=results/blockgtq_carekv; OUT=$RD/improved_ladder_v2.csv
if grep -q "^openlm-research/open_llama_7b_v2," "$OUT" 2>/dev/null; then echo "[olr] SKIP already in CSV"; exit 0; fi
echo "[olr] START openllama fine_rb8 SHARDED GPU 1,2 $(date +%m-%d_%H:%M)"
CUDA_VISIBLE_DEVICES=1,2 CAREKV_DEVICE_MAP=auto python run_bgtq_adapter.py \
  --model-id openlm-research/open_llama_7b_v2 --seq-len 512 --num-samples 32 \
  --append-csv "$OUT" > "$RD/ol_retry.log" 2>&1
echo "[olr] END rc=$? $(date +%m-%d_%H:%M) :: $(grep DONE $RD/ol_retry.log | tail -1)"
