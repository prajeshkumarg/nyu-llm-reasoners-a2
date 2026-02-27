#!/bin/bash
#SBATCH --job-name=bench_c
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/bench_c_%j.out
#SBATCH --error=./logs/bench_c_%j.err
#SBATCH --requeue

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"

mkdir -p "${REPO_DIR}/results"
OUT_C="${REPO_DIR}/results/bench_part_c.csv"
echo "model,mode,warmup_steps,avg_ms,std_ms,min_ms,max_ms" > "$OUT_C"

# Part c only needs small model — warmup effect is most visible there
# and running all 5 models x 3 warmup values x 2 modes = 30 runs (slow)
for MODEL in small; do
    for WARMUP in 0 1 2; do
        for FORWARD_FLAG in "" "--forward-only"; do
            if [ -z "$FORWARD_FLAG" ]; then
                MODE="forward-backward"
            else
                MODE="forward-only"
            fi

            echo ""
            echo "=========================================="
            echo "  model=$MODEL | warmup=$WARMUP | $MODE"
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
                    --warmup-steps ${WARMUP} \
                    --num-steps 10 \
                    ${FORWARD_FLAG}
              ") || {
                echo "  FAILED (likely OOM)"
                echo "${MODEL},${MODE},${WARMUP},OOM,OOM,OOM,OOM" >> "$OUT_C"  # fixed: $OUT_C
                continue
            }

            echo "$OUTPUT"

            AVG=$(echo "$OUTPUT" | grep 'Average step time' | sed 's/Average step time: \([0-9.]*\) ms.*/\1/')
            STD=$(echo "$OUTPUT" | grep 'Average step time' | sed 's/.*(std: \([0-9.]*\) ms).*/\1/')
            MIN=$(echo "$OUTPUT" | grep 'Min step time'     | sed 's/Min step time: *\([0-9.]*\) ms/\1/')
            MAX=$(echo "$OUTPUT" | grep 'Max step time'     | sed 's/Max step time: *\([0-9.]*\) ms/\1/')

            # fixed: write to $OUT_C, include WARMUP, drop MEM (not in header)
            echo "${MODEL},${MODE},${WARMUP},${AVG},${STD},${MIN},${MAX}" >> "$OUT_C"
        done
    done
done

echo ""
echo "=== Done! Results saved to $OUT_C ==="
cat "$OUT_C"