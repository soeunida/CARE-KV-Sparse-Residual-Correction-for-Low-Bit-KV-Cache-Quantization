#!/usr/bin/env bash
# run_rebuttal_split.sh — launch ONE independent rebuttal experiment on ONE GPU.
# Designed so you can open several terminals and run different (GPU, TASK) pairs
# in parallel without collision. Every task writes its OWN csv and is resumable
# (re-running skips rows already present).
#
# Usage:   bash scripts/run_rebuttal_split.sh <GPU> <TASK>
# Example: bash scripts/run_rebuttal_split.sh 2 mag512
#
# TASKS
#   mag512 mag1024 mag2048 mag4096   query-AGNOSTIC (residual-magnitude) CARE-KV
#                                    PG-19 N=2 at that SL  (scaling-figure control arm)
#   qa8192 mag8192                   qaware / magnitude CARE-KV PG-19 SL=8192 N=2
#   fast8192                         fp16/turbo/base PG-19 SL=8192 N=2 (if not done)
#   wt2                              WikiText-2 continuity: all modes, SL=1024 & 4096, N=2
#   ds_fast                          DeepSeek MMLU+ARC: fp16/base/turbo (fast)
#   ds_carekv                        DeepSeek MMLU+ARC: carekv only (slow)
#
# NOTE: check `nvidia-smi` first and pass a GPU whose memory.used is ~0.
set -u
GPU="${1:?need GPU index}"; TASK="${2:?need TASK}"
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun
export CAREKV_DUMP_IMPORTANCE=1 LC_CHUNK_SIZE=512
MI=mistralai/Mistral-7B-v0.3
DS=deepseek-ai/deepseek-llm-7b-base
SC=results/longctx_ppl/scaling
DM=results/downstream_mc
mkdir -p "$SC" "$DM"
run(){ echo "[split] GPU$GPU TASK=$TASK :: $*"; CUDA_VISIBLE_DEVICES="$GPU" "$@"; }

case "$TASK" in
  mag512|mag1024|mag2048|mag4096)
    SL="${TASK#mag}"
    run python tools/eval_longctx_ppl.py --model-id $MI --dataset pg19 \
      --seq-lens "$SL" --num-samples 2 --modes carekv_magnitude \
      --out-csv "$SC/mistral_mag_sl${SL}.csv" ;;
  qa8192)
    run python tools/eval_longctx_ppl.py --model-id $MI --dataset pg19 \
      --seq-lens 8192 --num-samples 2 --modes carekv_qaware \
      --out-csv "$SC/mistral_sl8192.csv" ;;
  mag8192)
    run python tools/eval_longctx_ppl.py --model-id $MI --dataset pg19 \
      --seq-lens 8192 --num-samples 2 --modes carekv_magnitude \
      --out-csv "$SC/mistral_mag_sl8192.csv" ;;
  fast8192)
    run python tools/eval_longctx_ppl.py --model-id $MI --dataset pg19 \
      --seq-lens 8192 --num-samples 2 --modes fp16 turboquant_int3 base_quant_int3 \
      --out-csv "$SC/mistral_fast_sl8192.csv" ;;
  wt2)
    run python tools/eval_longctx_ppl.py --model-id $MI --dataset wikitext \
      --seq-lens 1024 4096 --num-samples 2 \
      --modes fp16 base_quant_int3 turboquant_int3 carekv_qaware \
      --out-csv "$SC/mistral_wt2_n2.csv" ;;
  ds_fast)   # fast modes; separate CSV so it can run parallel to ds_carekv
    run python tools/eval_downstream_mc.py --model-id $DS \
      --modes fp16 base_quant_int3 turboquant_int3 --tasks mmlu arc \
      --mmlu-n 100 --arc-n 40 --out-csv "$DM/deepseek_fast.csv" ;;
  ds_carekv) # slow; matched n=100/40 for a fair comparison
    run python tools/eval_downstream_mc.py --model-id $DS \
      --modes carekv_stored_int3 --tasks mmlu arc \
      --mmlu-n 100 --arc-n 40 --out-csv "$DM/deepseek_carekv.csv" ;;
  *) echo "unknown TASK: $TASK"; exit 2 ;;
esac
echo "[split] DONE GPU$GPU TASK=$TASK"
