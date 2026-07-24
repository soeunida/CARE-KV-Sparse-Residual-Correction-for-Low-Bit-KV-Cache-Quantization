#!/usr/bin/env bash
# Re-run the REMAINING ladder models with the improved fine_rb8 config
# (kcg16 vtb2, 8/8/8/8, exact, 8-bit residual), SL512 N32. Single-GPU models run
# in parallel serial-queues; Yi-34B (4-GPU, device_map=auto) runs last once the
# single-GPU queues free their GPUs.
set -u
cd /home/soeun/care_kv_clean
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun TRANSFORMERS_VERBOSITY=error
RD=results/blockgtq_carekv
OUT=$RD/improved_ladder.csv
# fine_rb8 config (env consumed by run_bgtq_adapter.py)
export CAREKV_CFG_KCG=16 CAREKV_CFG_VTB=2 CAREKV_CFG_KMODE=exact CAREKV_CFG_RB=8
export CAREKV_CFG_SK=8 CAREKV_CFG_SV=8 CAREKV_CFG_RK=8 CAREKV_CFG_RV=8

run1 () {  # gpu model
  local gpu=$1 model=$2 tag
  tag=$(basename "$model")
  echo "[imp] >>> $tag START gpu$gpu $(date +%m-%d_%H:%M)"
  CUDA_VISIBLE_DEVICES=$gpu python run_bgtq_adapter.py --model-id "$model" \
    --seq-len 512 --num-samples 32 --append-csv "$OUT" \
    > "$RD/improved_${tag}.log" 2>&1
  echo "[imp] >>> $tag END gpu$gpu rc=$? $(date +%m-%d_%H:%M) :: $(grep 'DONE' "$RD/improved_${tag}.log" | grep -v '^  BlockGTQ:' | tail -1)"
}

# Parallel serial-queues on 3 single-GPU cards.
( run1 5 NousResearch/Llama-2-13b-hf ) &          # longest → alone
( run1 6 deepseek-ai/deepseek-llm-7b-base; run1 6 TinyLlama/TinyLlama-1.1B-Chat-v1.0 ) &
( run1 4 openlm-research/open_llama_7b_v2 ) &
wait
echo "[imp] single-GPU models DONE $(date +%m-%d_%H:%M) — starting Yi-34B (4-GPU)"

# Yi-34B: 4-GPU sharded (gc-fix in run_bgtq_adapter frees calib model first).
export CUDA_VISIBLE_DEVICES=1,4,5,6 CAREKV_DEVICE_MAP=auto
echo "[imp] >>> Yi-34B START $(date +%m-%d_%H:%M)"
python run_bgtq_adapter.py --model-id 01-ai/Yi-34B --seq-len 512 --num-samples 32 \
  --append-csv "$OUT" > "$RD/improved_Yi-34B.log" 2>&1
echo "[imp] >>> Yi-34B END rc=$? $(date +%m-%d_%H:%M) :: $(grep 'DONE' "$RD/improved_Yi-34B.log" | grep -v '^  BlockGTQ:' | tail -1)"
echo "[imp] ALL DONE $(date +%m-%d_%H:%M)"
