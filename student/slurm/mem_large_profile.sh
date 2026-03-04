#!/bin/bash
#SBATCH --job-name=mem_profile_large
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/mem_profile_large_%j.out
#SBATCH --error=./logs/mem_profile_large_%j.err
#SBATCH --requeue

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"
MEM_DIR="${REPO_DIR}/results/memory"

mkdir -p "${MEM_DIR}"
mkdir -p "${REPO_DIR}/logs"

SCRIPT="student/memory_nsys_profiling.py"

echo "========================================"
echo "  Memory profiling sweep (large model)"
echo "  Goal: find largest model where training"
echo "  does not OOM across all context lengths"
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
# large model — forward + training across all context lengths
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== large: forward + training, all context lengths ==="

for CTX in 128 256 512 1024; do
    run_mem "large" ${CTX} "--forward-only" "large ctx=${CTX} forward-only"
    run_mem "large" ${CTX} ""               "large ctx=${CTX} full training step"
done

# ════════════════════════════════════════════════════════════════════════════════
# Part (c): large mixed-precision for comparison
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== large: mixed-precision ctx=128 ==="

run_mem "large" 128 "--forward-only --mixed-precision" "large ctx=128 bf16 forward-only"
run_mem "large" 128 "--mixed-precision"                "large ctx=128 bf16 full training step"

# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Done. Results:"
echo "========================================"
echo ""
echo "Peak memory summary (grep from log):"
grep "peak GPU memory" ./logs/mem_profile_large_${SLURM_JOB_ID}.out 2>/dev/null || \
    grep "\[mem\] peak" ./logs/mem_profile_large_${SLURM_JOB_ID}.out 2>/dev/null || \
    echo "  (check log file manually)"
echo ""
ls -lh "${MEM_DIR}"/memory_large*.pickle 2>/dev/null || echo "  (no large pickles found)"
echo ""
echo "scp pg2973@greene.hpc.nyu.edu:${MEM_DIR}/memory_large*.pickle ."