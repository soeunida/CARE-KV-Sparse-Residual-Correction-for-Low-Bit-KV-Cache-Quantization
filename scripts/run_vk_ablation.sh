#!/usr/bin/env bash
set -euo pipefail

cd /home/soeun/CARE_KV/care_kv
mkdir -p results

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODEL_ID=${MODEL_ID:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
export PYTHONPATH=${PYTHONPATH:-/home/soeun}

SEQ_LIST="${SEQ_LIST:-128}"
BITS_LIST="${BITS_LIST:-3 2}"
R="${R:-0.05}"
K_LIST="${K_LIST:-0.02 0.05 0.10}"

for L in $SEQ_LIST; do
  for B in $BITS_LIST; do
    echo "===== BASE_QUANT INT$B SEQ_LEN=$L ====="
    MODE=carekv \
    CAREKV_RETURN=care \
    CAREKV_PREFILL_MODE=base_quant \
    BASE_BITS=$B \
    SEQ_LEN=$L \
    python eval_ppl_simple.py | tee results/ppl_ablation_basequant_int${B}_${L}.txt

    echo "===== V-only INT$B SEQ_LEN=$L R=$R ====="
    MODE=carekv \
    CAREKV_RETURN=care \
    CAREKV_PREFILL_MODE=carekv \
    CAREKV_PREFILL_RESIDUAL_KIND=v \
    BASE_BITS=$B \
    CAREKV_PREFILL_RESIDUAL_RATIO=$R \
    CAREKV_K_CORRECTION_SCALE=0.0 \
    SEQ_LEN=$L \
    python eval_ppl_simple.py | tee results/ppl_ablation_int${B}_${L}_Vonly_R${R}.txt

    for K in $K_LIST; do
      echo "===== K-only INT$B SEQ_LEN=$L R=$R K=$K ====="
      MODE=carekv \
      CAREKV_RETURN=care \
      CAREKV_PREFILL_MODE=carekv \
      CAREKV_PREFILL_RESIDUAL_KIND=k \
      BASE_BITS=$B \
      CAREKV_PREFILL_RESIDUAL_RATIO=$R \
      CAREKV_K_CORRECTION_SCALE=$K \
      SEQ_LEN=$L \
      python eval_ppl_simple.py | tee results/ppl_ablation_int${B}_${L}_Konly_R${R}_K${K}.txt

      echo "===== V+K INT$B SEQ_LEN=$L R=$R K=$K ====="
      MODE=carekv \
      CAREKV_RETURN=care \
      CAREKV_PREFILL_MODE=carekv \
      CAREKV_PREFILL_RESIDUAL_KIND=both \
      BASE_BITS=$B \
      CAREKV_PREFILL_RESIDUAL_RATIO=$R \
      CAREKV_K_CORRECTION_SCALE=$K \
      SEQ_LEN=$L \
      python eval_ppl_simple.py | tee results/ppl_ablation_int${B}_${L}_VK_R${R}_K${K}.txt
    done
  done
done

echo "===== SUMMARY ====="
python tools/summarize_ppl_results.py results/ppl_ablation_*.txt
