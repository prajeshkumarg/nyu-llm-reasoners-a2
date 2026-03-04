#!/bin/bash
#SBATCH --job-name=mem_profile
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/mem_profile_%j.out
#SBATCH --error=./logs/mem_profile_%j.err
#SBATCH --requeue

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"
MEM_DIR="${REPO_DIR}/results/memory"

mkdir -p "${MEM_DIR}"
mkdir -p "${REPO_DIR}/logs"

SCRIPT="student/memory_nsys_profiling.py"

echo "========================================"
echo "  Memory profiling sweep (xl model)"
echo "  output: ${MEM_DIR}"
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
# Part (a): xl forward-only + xl full training step at ctx=128
# Deliverable: two memory timeline images
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (a): xl forward + train timelines ==="

run_mem "xl" 128 "--forward-only" "xl ctx=128 forward-only"
run_mem "xl" 128 ""               "xl ctx=128 full training step"

# ════════════════════════════════════════════════════════════════════════════════
# Part (b): xl peak memory across context lengths (forward + train)
# Deliverable: table with two numbers per context length
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (b): xl peak memory across context lengths ==="

for CTX in 128 256 512 1024; do
    run_mem "xl" ${CTX} "--forward-only" "xl ctx=${CTX} forward-only"
    run_mem "xl" ${CTX} ""               "xl ctx=${CTX} full training step"
done

# ════════════════════════════════════════════════════════════════════════════════
# Part (c): xl mixed-precision memory (forward + train)
# Deliverable: compare fp32 vs bf16 peak memory
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (c): xl mixed-precision memory ==="

run_mem "xl" 128 "--forward-only --mixed-precision" "xl ctx=128 bf16 forward-only"
run_mem "xl" 128 "--mixed-precision"                "xl ctx=128 bf16 full training step"

# ════════════════════════════════════════════════════════════════════════════════
# Part (e): reuse part (a) forward pickle for stack trace inspection
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (e): reuse xl ctx128 forward pickle from part (a) ==="
echo "  Load memory_xl_ctx128_fp32_fwd.pickle into https://pytorch.org/memory_viz"
echo "  Set Detail slider to 10% to find largest allocations + stack traces"

# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Done. Pickle files saved to ${MEM_DIR}"
echo "========================================"
ls -lh "${MEM_DIR}"/*.pickle 2>/dev/null || echo "  (none found)"
echo ""
echo "scp pg2973@greene.hpc.nyu.edu:${MEM_DIR}/*.pickle ."