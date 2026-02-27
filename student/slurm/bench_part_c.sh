#!/bin/bash
#SBATCH --job-name=bench_c
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/bench_c_%j.out
#SBATCH --error=./logs/bench_c_%j.err
#SBATCH --requeue

source "$(dirname "$0")/setup.sh"

RESULTS_DIR="${REPO_DIR}/results"
mkdir -p "${RESULTS_DIR}"
OUT="${RESULTS_DIR}/bench_part_c.csv"
echo "warmup,mode,avg_ms,std_ms,min_ms,max_ms" > "$OUT"

echo "=== Part (c): Warmup ablation on 'small' model, ctx=128 ==="

for W in 0 1 2 5; do
    for FORWARD_FLAG in "" "--forward-only"; do
        if [ -z "$FORWARD_FLAG" ]; then
            MODE="forward-backward"
        else
            MODE="forward-only"
        fi

        echo ""
        echo "=== warmup=$W | $MODE ==="

        OUTPUT=$(run_in_container "uv run python student/basicprofiling.py \
            --model-size small \
            --context-length 128 \
            --warmup-steps ${W} \
            --num-steps 10 \
            ${FORWARD_FLAG}")

        echo "$OUTPUT"

        AVG=$(echo "$OUTPUT" | grep 'Average' | sed 's/.*: \([0-9.]*\) ms.*/\1/')
        STD=$(echo "$OUTPUT" | grep 'Average' | sed 's/.*(std: \([0-9.]*\) ms).*/\1/')
        MIN=$(echo "$OUTPUT" | grep 'Min' | sed 's/.*: *\([0-9.]*\) ms/\1/')
        MAX=$(echo "$OUTPUT" | grep 'Max' | sed 's/.*: *\([0-9.]*\) ms/\1/')

        echo "${W},${MODE},${AVG},${STD},${MIN},${MAX}" >> "$OUT"
    done
done

echo ""
echo "Results written to $OUT"
