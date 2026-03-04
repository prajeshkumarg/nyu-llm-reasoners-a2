#!/bin/bash
#SBATCH --job-name=mem_profile_medium
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/mem_profile_medium_%j.out
#SBATCH --error=./logs/mem_profile_medium_%j.err
#SBATCH --requeue

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"
MEM_DIR="${REPO_DIR}/results/memory"

mkdir -p "${MEM_DIR}"
mkdir -p "${REPO_DIR}/logs"

SCRIPT="student/memory_nsys_profiling.py"

echo "========================================"
echo "  Memory profiling sweep (medium model)"
echo "========================================"

run_mem() {
    local MODEL=$1
    local CTX=$2
    local EXTRA_FLAGS=$3
    local LABEL=$4

    echo ""
    echo "------------------------------------------"
    echo "  ${LABEL}"
    echo "------------------------------------------"

    if ! singularity exec --nv \
      --overlay "${OVERLAY}:ro" \
      "${SIF}" \
      /bin/bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        cd ${REPO_DIR}
        uv run python ${SCRIPT} \
          --model-size ${MODEL} \
          --context-length ${CTX} \
          --warmup-steps 2 \
          --memory-profile \
          --memory-output-dir ${MEM_DIR} \
          ${EXTRA_FLAGS}
      "; then
        echo "  FAILED / OOM: ${LABEL}"
    else
        echo "  OK: ${LABEL}"
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# forward + training across all context lengths
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== medium: forward + training, all context lengths ==="

for CTX in 128 256 512 1024; do
    run_mem "medium" ${CTX} "--forward-only" "medium ctx=${CTX} forward-only"
    run_mem "medium" ${CTX} ""               "medium ctx=${CTX} full training step"
done

# ════════════════════════════════════════════════════════════════════════════════
# mixed-precision at ctx=128
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== medium: mixed-precision ctx=128 ==="

run_mem "medium" 128 "--forward-only --mixed-precision" "medium ctx=128 bf16 forward-only"
run_mem "medium" 128 "--mixed-precision"                "medium ctx=128 bf16 full training step"

# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Done."
echo "========================================"
ls -lh "${MEM_DIR}"/memory_medium*.pickle 2>/dev/null || echo "  (no medium pickles found)"
echo ""
echo "scp pg2973@greene.hpc.nyu.edu:${MEM_DIR}/memory_medium*.pickle ."