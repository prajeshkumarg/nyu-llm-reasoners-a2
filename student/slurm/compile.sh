#!/bin/bash
#SBATCH --job-name=attn_compile
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/attn_compile_%j.out
#SBATCH --error=./logs/attn_compile_%j.err

OVERLAY="/scratch/pg2973/overlay-25GB-500K.ext3"
SIF="/scratch/pg2973/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/pg2973/nyu-llm-reasoners-a2"

mkdir -p ${REPO_DIR}/logs

run() {
    singularity exec --nv --overlay "${OVERLAY}:ro" "${SIF}" \
      /bin/bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        cd ${REPO_DIR}
        uv run python student/attention_bench.py $1 --output $2
      "
}

echo "=== eager ==="
run "" results/attention_eager.csv

echo "=== compiled ==="
run "--compiled" results/attention_compiled.csv