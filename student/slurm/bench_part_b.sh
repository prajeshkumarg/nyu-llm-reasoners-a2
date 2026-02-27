#!/bin/bash
#SBATCH --job-name=bench_b
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/bench_b_%j.out
#SBATCH --error=./logs/bench_b_%j.err
#SBATCH --requeue

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"

mkdir -p "${REPO_DIR}/results"
OUT="${REPO_DIR}/results/bench_part_b.csv"
echo "model,mode,avg_ms,std_ms,min_ms,max_ms,peak_mem_mb" > "$OUT"

for MODEL in small medium large xl 2.7B; do
    for FORWARD_FLAG in "" "--forward-only"; do
        if [ -z "$FORWARD_FLAG" ]; then
            MODE="forward-backward"
        else
            MODE="forward-only"
        fi

        echo ""
        echo "=========================================="
        echo "  model=$MODEL | $MODE"
        echo "=========================================="

        OUTPUT=$(singularity exec --bind /scratch --nv \
          --overlay "${OVERLAY}:ro" \
          "${SIF}" \
          /bin/bash -c "
            export PATH=\$HOME/.local/bin:\$PATH
            set -euo pipefail
            cd ${REPO_DIR}
            uv run python student/basicprofiling.py \
                --model-size ${MODEL} \
                --warmup-steps 5 \
                --num-steps 10 \
                ${FORWARD_FLAG}
          ") || {
            echo "  FAILED (likely OOM)"
            echo "${MODEL},${MODE},OOM,OOM,OOM,OOM,OOM" >> "$OUT"
            continue
        }

        echo "$OUTPUT"

        # Fix: anchor regex to label name so greedy .* can't overshoot to std:
        AVG=$(echo "$OUTPUT" | grep 'Average step time' | sed 's/Average step time: \([0-9.]*\) ms.*/\1/')
        STD=$(echo "$OUTPUT" | grep 'Average step time' | sed 's/.*(std: \([0-9.]*\) ms).*/\1/')
        MIN=$(echo "$OUTPUT" | grep 'Min step time'     | sed 's/Min step time: *\([0-9.]*\) ms/\1/')
        MAX=$(echo "$OUTPUT" | grep 'Max step time'     | sed 's/Max step time: *\([0-9.]*\) ms/\1/')
        MEM=$(echo "$OUTPUT" | grep 'Peak GPU memory'   | sed 's/Peak GPU memory: *\([0-9.]*\) MB/\1/')

        echo "${MODEL},${MODE},${AVG},${STD},${MIN},${MAX},${MEM}" >> "$OUT"
    done
done

echo ""
echo "=== Done! Results saved to $OUT ==="
cat "$OUT"