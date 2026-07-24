#!/usr/bin/env bash
# Re-run Yi-34B under fine_rb8 (config-consistent with the other 6 fine_rb8 rows).
# Yi-34B is GQA (nkv=8 → 60×8=480 layer×head units), NOT full-MHA like 13b, so the
# heavier exact/rk8-rv8 correction is feasible (cur_rb8 C-arm was only 68min).
# GPU 1,2,3 (34B needs 3), in parallel with openllama on GPU 0.
# Resumable: skips if Yi-34B already in the CSV (the cur_rb8 row was pruned first).
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
if grep -q "^01-ai/Yi-34B," "$OUT" 2>/dev/null; then echo "[yi] SKIP already in CSV"; exit 0; fi
echo "[yi] START fine_rb8 $(date +%m-%d_%H:%M) — GPU 1,2,3"
CUDA_VISIBLE_DEVICES=1,2,3 CAREKV_DEVICE_MAP=auto python run_bgtq_adapter.py \
  --model-id 01-ai/Yi-34B --seq-len 512 --num-samples 32 --append-csv "$OUT" \
  > "$RD/yi34b_fine.log" 2>&1
echo "[yi] END rc=$? $(date +%m-%d_%H:%M) :: $(grep DONE $RD/yi34b_fine.log | tail -1)"
