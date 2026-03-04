#!/bin/bash
#SBATCH --job-name=attn_bench
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/attn_bench_%j.out
#SBATCH --error=./logs/attn_bench_%j.err

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"

singularity exec --nv \
  --overlay "${OVERLAY}:ro" \
  "${SIF}" \
  /bin/bash -c "
    export PATH=\$HOME/.local/bin:\$PATH
    cd ${REPO_DIR}
    uv run python student/attention_bench.py \
      --output results/attention_bench.csv
  "