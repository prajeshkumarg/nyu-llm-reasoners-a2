#!/bin/bash
#SBATCH --job-name=mem_profile
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/mem_profile_%j.out
#SBATCH --error=./logs/mem_profile_%j.err
#SBATCH --requeue

# ── paths ──────────────────────────────────────────────────────────────────────
OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"
MEM_DIR="${REPO_DIR}/results/memory"

mkdir -p "${MEM_DIR}"
mkdir -p "${REPO_DIR}/logs"

SCRIPT="student/memory_nsys_profiling.py"

echo "========================================"
echo "  Memory profiling sweep"
echo "  output: ${MEM_DIR}"
echo "========================================"

# ── helper ────────────────────────────────────────────────────────────────────
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
# Part (a): 2.7B forward-only and full training step at ctx=128
# Deliverable: two memory timeline images
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (a): 2.7B forward vs train timelines ==="

run_mem "2.7B" 128 "--forward-only" "2.7B ctx=128 forward-only"
run_mem "2.7B" 128 ""               "2.7B ctx=128 full training step"

# ════════════════════════════════════════════════════════════════════════════════
# Part (b): 2.7B peak memory across context lengths (forward + train)
# Deliverable: table of peak memory per context length
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (b): 2.7B peak memory across context lengths ==="

for CTX in 128 256 512; do
    run_mem "2.7B" ${CTX} "--forward-only" "2.7B ctx=${CTX} forward-only"
    run_mem "2.7B" ${CTX} ""               "2.7B ctx=${CTX} full training step"
done

# ctx=1024 likely OOM for 2.7B train — try forward only
run_mem "2.7B" 1024 "--forward-only" "2.7B ctx=1024 forward-only (may OOM)"

# ════════════════════════════════════════════════════════════════════════════════
# Part (c): mixed-precision memory usage (forward + train)
# Deliverable: compare fp32 vs bf16 peak memory
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (c): 2.7B mixed-precision memory ==="

run_mem "2.7B" 128 "--forward-only --mixed-precision" "2.7B ctx=128 bf16 forward-only"
run_mem "2.7B" 128 "--mixed-precision"                "2.7B ctx=128 bf16 full training step"

# ════════════════════════════════════════════════════════════════════════════════
# Part (e): high-detail forward snapshot for stack trace inspection
# Same as part (a) forward — already generated above, reuse that pickle
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Part (e): already covered by part (a) forward snapshot ==="
echo "  Load memory_2.7B_ctx128_fwd.pickle into https://pytorch.org/memory_viz"
echo "  Reduce Detail slider to 10% to find largest allocations + stack traces"

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════
echo ""
echo "========================================"
echo "  Done. Pickle files saved to ${MEM_DIR}"
echo "========================================"
echo ""
echo "  Files generated:"
ls -lh "${MEM_DIR}"/*.pickle 2>/dev/null || echo "  (none found)"
echo ""
echo "  To download to local machine:"
echo "  scp pg2973@greene.hpc.nyu.edu:${MEM_DIR}/*.pickle ."
echo ""
echo "  Then load at: https://pytorch.org/memory_viz"