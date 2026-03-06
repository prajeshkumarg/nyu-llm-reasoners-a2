#!/bin/bash
#SBATCH --job-name=flash_bench
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/flash_bench_%j.out
#SBATCH --error=./logs/flash_bench_%j.err

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"

mkdir -p ${REPO_DIR}/logs

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -c "
    export PATH=\$HOME/.local/bin:\$PATH
    cd ${REPO_DIR}
    uv run python student/flash_bench.py
  "