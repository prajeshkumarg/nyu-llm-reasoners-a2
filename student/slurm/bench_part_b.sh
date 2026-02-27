#!/bin/bash
#SBATCH --job-name=bench_b
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/bench_b_%A_%a.out
#SBATCH --error=./logs/bench_b_%A_%a.err
#SBATCH --requeue
#SBATCH --array=0-9   # 5 models x 4 ctx_lens x 2 modes = 40 jobs

source "$(dirname "$0")/setup.sh"

MODEL_SIZES=(small medium large xl 2.7B)
CTX_LENS=(128 256 512 1024)
MODES=(forward-only forward-backward)

n_ctx=${#CTX_LENS[@]}
n_mode=${#MODES[@]}

model_idx=$(( SLURM_ARRAY_TASK_ID / (n_ctx * n_mode) ))
remainder=$(( SLURM_ARRAY_TASK_ID % (n_ctx * n_mode) ))
ctx_idx=$(( remainder / n_mode ))
mode_idx=$(( remainder % n_mode ))

MODEL=${MODEL_SIZES[$model_idx]}
CTX=${CTX_LENS[$ctx_idx]}
MODE=${MODES[$mode_idx]}

FORWARD_FLAG=""
if [ "$MODE" = "forward-only" ]; then
    FORWARD_FLAG="--forward-only"
fi

RESULTS_DIR="${REPO_DIR}/results/bench_b"
mkdir -p "${RESULTS_DIR}"

echo "=== Job $SLURM_ARRAY_TASK_ID: model=$MODEL ctx=$CTX mode=$MODE ==="

OUTPUT=$(run_in_container "uv run python student/basicprofiling.py \
    --model-size ${MODEL} \
    --context-length ${CTX} \
    --warmup-steps 5 \
    --num-steps 10 \
    ${FORWARD_FLAG}")

echo "$OUTPUT"

# Parse and store one CSV row per task
AVG=$(echo "$OUTPUT" | grep 'Average' | sed 's/.*: \([0-9.]*\) ms.*/\1/')
STD=$(echo "$OUTPUT" | grep 'Average' | sed 's/.*(std: \([0-9.]*\) ms).*/\1/')
MIN=$(echo "$OUTPUT" | grep 'Min' | sed 's/.*: *\([0-9.]*\) ms/\1/')
MAX=$(echo "$OUTPUT" | grep 'Max' | sed 's/.*: *\([0-9.]*\) ms/\1/')

echo "${MODEL},${CTX},${MODE},${AVG},${STD},${MIN},${MAX}" \
  > "${RESULTS_DIR}/${SLURM_ARRAY_TASK_ID}.csv"

echo "Result saved to ${RESULTS_DIR}/${SLURM_ARRAY_TASK_ID}.csv"
