#!/bin/bash
#SBATCH --job-name=nsys_profile
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=./logs/nsys_profile_%j.out
#SBATCH --error=./logs/nsys_profile_%j.err
#SBATCH --requeue

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"
OUT_DIR="${REPO_DIR}/results/nsys"
NSYS="/home/pg2973/tools/nsight-systems/pkg/bin/nsys"

mkdir -p "${OUT_DIR}"
mkdir -p "${REPO_DIR}/logs"

# ── helper: run inside container ───────────────────────────────────────────────
# --bind /scratch is REMOVED — /scratch is already auto-mounted by the cluster,
# keeping it caused: WARNING: destination is already in the mount point list
container_exec() {
    singularity exec --nv \
      --overlay "${OVERLAY}:ro" \
      "${SIF}" \
      /bin/bash -c "export PATH=\$HOME/.local/bin:\$PATH; $*"
}

# ── pre-flight checks ──────────────────────────────────────────────────────────
echo "========================================"
echo "  Pre-flight checks"
echo "========================================"

echo -n "  nsys inside container... "
if ! container_exec "${NSYS} --version" > /dev/null 2>&1; then
    echo "FAIL"
    echo ""
    echo "  ERROR: nsys not found at: ${NSYS}"
    echo "  Find the correct path by running inside the container:"
    echo "    find / -name 'nsys' 2>/dev/null"
    echo "  Then update the NSYS variable at the top of this script."
    exit 1
fi
NSYS_VER=$(container_exec "${NSYS} --version" 2>&1 | head -1)
echo "OK  (${NSYS_VER})"

echo -n "  Python script exists... "
if [ ! -f "${REPO_DIR}/student/nsys_profiling.py" ]; then
    echo "FAIL"
    echo "  ERROR: ${REPO_DIR}/student/nsys_profiling.py not found."
    exit 1
fi
echo "OK"

echo -n "  Output directory writable... "
if ! touch "${OUT_DIR}/.write_test" 2>/dev/null; then
    echo "FAIL"
    echo "  ERROR: Cannot write to ${OUT_DIR}"
    exit 1
fi
rm -f "${OUT_DIR}/.write_test"
echo "OK"
echo ""

# ── sweep ──────────────────────────────────────────────────────────────────────
MODEL_SIZES=(small medium large xl 2.7B)
CONTEXT_LENGTHS=(128 256 512 1024)

echo "========================================"
echo "  nsys profiling sweep"
echo "  nsys : ${NSYS}"
echo "  out  : ${OUT_DIR}"
echo "========================================"

for MODEL in "${MODEL_SIZES[@]}"; do
  for CTX in "${CONTEXT_LENGTHS[@]}"; do
    for MODE_FLAG in "--forward-only" ""; do

      if [ -z "$MODE_FLAG" ]; then MODE_LABEL="train"; else MODE_LABEL="fwd"; fi

      LABEL="${MODEL}_ctx${CTX}_${MODE_LABEL}"
      OUT_FILE="${OUT_DIR}/${LABEL}"

      HOOK_FLAG=""
      if [[ "$MODEL" == "xl" || "$MODEL" == "2.7B" ]]; then
          HOOK_FLAG="--no-layer-hooks"
      fi

      echo ""
      echo "------------------------------------------"
      echo "  model=${MODEL}  ctx=${CTX}  mode=${MODE_LABEL}"
      echo "  output: ${OUT_FILE}.nsys-rep"
      echo "------------------------------------------"

      # Note: no set -euo pipefail inside bash -c —
      # nsys sometimes exits non-zero even on success and set -e would
      # misinterpret that as a failure and kill the run.
      if ! singularity exec --nv \
        --overlay "${OVERLAY}:ro" \
        "${SIF}" \
        /bin/bash -c "
          export PATH=\$HOME/.local/bin:\$PATH
          cd ${REPO_DIR}
          ${NSYS} profile \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            --trace=cuda,nvtx \
            --force-overwrite=true \
            -o ${OUT_FILE} \
            uv run python student/nsys_profiling.py \
              --model-size ${MODEL} \
              --context-length ${CTX} \
              --warmup-steps 3 \
              ${HOOK_FLAG} \
              ${MODE_FLAG}
        "; then
          echo "  FAILED / OOM: ${LABEL}"
          continue
      fi

      echo "  OK: ${LABEL}"
    done
  done
done

echo ""
echo "========================================"
echo "  Sweep done. Generating stats..."
echo "========================================"

# ── 1. CUDA GPU Kernel Summary (questions a/b/c/d) ─────────────────────────────
for REP in "${OUT_DIR}"/*.nsys-rep; do
    [ -f "$REP" ] || continue
    STEM="${REP%.nsys-rep}"
    STATS_DIR="${STEM}_kern_stats"
    mkdir -p "${STATS_DIR}"
    if ! singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" \
      /bin/bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        ${NSYS} stats --report cuda_gpu_kern_sum --format csv \
            --output ${STATS_DIR} ${REP}
      "; then
        echo "  kern stats FAILED: $(basename ${STEM})"
    else
        echo "  kern stats OK: $(basename ${STEM})"
    fi
done

# ── 2. NVTX Summary (question e: attn/softmax vs attn/qk_matmul) ───────────────
for REP in "${OUT_DIR}"/*_fwd.nsys-rep; do
    [ -f "$REP" ] || continue
    STEM="${REP%.nsys-rep}"
    NVTX_DIR="${STEM}_nvtx_stats"
    mkdir -p "${NVTX_DIR}"
    if ! singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" \
      /bin/bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        ${NSYS} stats --report nvtx_sum --format csv \
            --output ${NVTX_DIR} ${REP}
      "; then
        echo "  nvtx stats FAILED: $(basename ${STEM})"
    else
        echo "  nvtx stats OK: $(basename ${STEM})"
    fi
done

echo ""
echo "========================================"
echo "  All done."
ls -lh "${OUT_DIR}"/*.nsys-rep 2>/dev/null || echo "  (no .nsys-rep files found)"
echo "========================================"