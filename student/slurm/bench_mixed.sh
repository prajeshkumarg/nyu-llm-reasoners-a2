#!/bin/bash
#SBATCH --job-name=bench_mixed
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/bench_mixed_%j.out
#SBATCH --error=./logs/bench_mixed_%j.err
#SBATCH --requeue

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"

mkdir -p "${REPO_DIR}/results"
OUT="${REPO_DIR}/results/bench_mixed_precision.csv"
echo "model,precision,mode,avg_ms,std_ms,min_ms,max_ms,peak_mem_mb" > "$OUT"

for MODEL in small medium large xl 2.7B; do
  for PRECISION_FLAG in "" "--mixed-precision"; do
    for FORWARD_FLAG in "" "--forward-only"; do

      if [ -z "$PRECISION_FLAG" ]; then PRECISION="fp32"; else PRECISION="bf16"; fi
      if [ -z "$FORWARD_FLAG" ];   then MODE="forward-backward"; else MODE="forward-only"; fi

      echo ""
      echo "=========================================="
      echo "  model=$MODEL | precision=$PRECISION | $MODE"
      echo "=========================================="

      OUTPUT=$(singularity exec --nv \
        --overlay "${OVERLAY}:ro" \
        "${SIF}" \
        /bin/bash -c "
          export PATH=\$HOME/.local/bin:\$PATH
          cd ${REPO_DIR}
          uv run python student/bfprofiling.py \
              --model-size ${MODEL} \
              --warmup-steps 5 \
              --num-steps 10 \
              ${PRECISION_FLAG} \
              ${FORWARD_FLAG}
        ") || {
          echo "  FAILED (likely OOM)"
          echo "${MODEL},${PRECISION},${MODE},OOM,OOM,OOM,OOM,OOM" >> "$OUT"
          continue
      }

      echo "$OUTPUT"

      AVG=$(echo "$OUTPUT" | grep 'Average step time' | sed 's/Average step time: \([0-9.]*\) ms.*/\1/')
      STD=$(echo "$OUTPUT" | grep 'Average step time' | sed 's/.*(std: \([0-9.]*\) ms).*/\1/')
      MIN=$(echo "$OUTPUT" | grep 'Min step time'     | sed 's/Min step time: *\([0-9.]*\) ms/\1/')
      MAX=$(echo "$OUTPUT" | grep 'Max step time'     | sed 's/Max step time: *\([0-9.]*\) ms/\1/')
      MEM=$(echo "$OUTPUT" | grep 'Peak GPU memory'   | sed 's/Peak GPU memory: *\([0-9.]*\) MB/\1/')

      echo "${MODEL},${PRECISION},${MODE},${AVG},${STD},${MIN},${MAX},${MEM}" >> "$OUT"

    done
  done
done

echo ""
echo "=== Done! Results saved to $OUT ==="
cat "$OUT"