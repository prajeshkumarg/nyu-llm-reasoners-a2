#!/bin/bash
#SBATCH --job-name=nsys_profile
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=./logs/nsys_profile_%j.out
#SBATCH --error=./logs/nsys_profile_%j.err
#SBATCH --requeue

# ── paths ──────────────────────────────────────────────────────────────────────
OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"
OUT_DIR="${REPO_DIR}/results/nsys"

mkdir -p "${OUT_DIR}"
mkdir -p "${REPO_DIR}/logs"

# ── sweep parameters ───────────────────────────────────────────────────────────
MODEL_SIZES=(small medium large xl 2.7B)
CONTEXT_LENGTHS=(128 256 512 1024)

echo "========================================"
echo "  nsys profiling sweep"
echo "  repo   : ${REPO_DIR}"
echo "  output : ${OUT_DIR}"
echo "========================================"

for MODEL in "${MODEL_SIZES[@]}"; do
  for CTX in "${CONTEXT_LENGTHS[@]}"; do
    for MODE_FLAG in "--forward-only" ""; do

      # Derive a readable label for filenames
      if [ -z "$MODE_FLAG" ]; then
          MODE_LABEL="train"
      else
          MODE_LABEL="fwd"
      fi

      LABEL="${MODEL}_ctx${CTX}_${MODE_LABEL}"
      OUT_FILE="${OUT_DIR}/${LABEL}"

      

      # Suppress per-layer hooks for large models — reduces trace clutter
      HOOK_FLAG=""
      if [[ "$MODEL" == "xl" || "$MODEL" == "2.7B" ]]; then
          HOOK_FLAG="--no-layer-hooks"
      fi

      echo ""
      echo "------------------------------------------"
      echo "  model=${MODEL}  ctx=${CTX}  mode=${MODE_LABEL}"
      echo "  output: ${OUT_FILE}.nsys-rep"
      echo "------------------------------------------"

      if ! singularity exec --bind /scratch --nv \
        --overlay "${OVERLAY}:ro" \
        "${SIF}" \
        /bin/bash -c "
          export PATH=\$HOME/.local/bin:\$PATH
          set -euo pipefail
          cd ${REPO_DIR}

          nsys profile \
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
echo "  Sweep done. Generating stats summaries..."
echo "========================================"

# ── 1. CUDA GPU Kernel Summary (all questions) ─────────────────────────────────
for REP in "${OUT_DIR}"/*.nsys-rep; do
    [ -f "$REP" ] || continue
    STEM="${REP%.nsys-rep}"
    STATS_DIR="${STEM}_kern_stats"
    mkdir -p "${STATS_DIR}"

    if ! singularity exec --bind /scratch --nv \
      --overlay "${OVERLAY}:ro" \
      "${SIF}" \
      /bin/bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        nsys stats \
            --report cuda_gpu_kern_sum \
            --format csv \
            --output ${STATS_DIR} \
            ${REP}
      "; then
        echo "  kern stats FAILED: $(basename ${STEM})"
    else
        echo "  kern stats OK: $(basename ${STEM})"
    fi
done

# ── 2. NVTX Range Summary (for question e: softmax vs matmul in attention) ─────
for REP in "${OUT_DIR}"/*_fwd.nsys-rep; do
    [ -f "$REP" ] || continue
    STEM="${REP%.nsys-rep}"
    NVTX_DIR="${STEM}_nvtx_stats"
    mkdir -p "${NVTX_DIR}"

    if ! singularity exec --bind /scratch --nv \
      --overlay "${OVERLAY}:ro" \
      "${SIF}" \
      /bin/bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        nsys stats \
            --report nvtx_sum \
            --format csv \
            --output ${NVTX_DIR} \
            ${REP}
      "; then
        echo "  nvtx stats FAILED: $(basename ${STEM})"
    else
        echo "  nvtx stats OK: $(basename ${STEM})"
    fi
done

echo ""
echo "========================================"
echo "  All done. Reports saved to ${OUT_DIR}"
echo "  .nsys-rep files:"
ls -lh "${OUT_DIR}"/*.nsys-rep 2>/dev/null || echo "  (none found)"
echo "========================================"