#!/usr/bin/env bash
# Per-arm isolated re-run for a big model: each arm in its own process (fresh
# memory → no device_map CPU-offload from accumulation). Args: MODEL_ID GPUS OUT_CSV
set -u
MODEL_ID="$1"; GPUS="$2"; OUT="$3"
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun CAREKV_DEBUG_STATS=1
export CUDA_VISIBLE_DEVICES="$GPUS"
# >1 GPU in the set → shard (auto); single GPU → default
if [[ "$GPUS" == *,* ]]; then export CAREKV_DEVICE_MAP=auto; else unset CAREKV_DEVICE_MAP; fi
rm -f "$OUT"
echo "[rerun] $(date '+%T') $MODEL_ID on GPU=$GPUS dmap=${CAREKV_DEVICE_MAP:-single}"
for arm in fp16 base_int3 carekv; do
  echo "[rerun] --- arm=$arm ---"
  python tools/eval_7b_validation.py --model-id "$MODEL_ID" --out-csv "$OUT" \
    --seq-len 512 --num-samples 4 --arm "$arm"
done
echo "[rerun] $(date '+%T') DONE $MODEL_ID"
