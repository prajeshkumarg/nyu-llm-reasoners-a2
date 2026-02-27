#!/bin/bash
# Shared setup sourced by all benchmark sbatch scripts.
# -------------------------------------------------------
NETID="pg2973"
OVERLAY="/scratch/${NETID}/overlay-25GB-500K.ext3"
SIF="/scratch/${NETID}/ubuntu-20.04.3.sif"
REPO_DIR="/scratch/${NETID}/nyu-llm-reasoners-a2"
# -------------------------------------------------------

# Helper: run a command inside singularity with uv available
run_in_container() {
    singularity exec --bind /scratch --nv \
      --overlay "${OVERLAY}:ro" \
      "${SIF}" \
      /bin/bash -c "
        export PATH=\$HOME/.local/bin:\$PATH
        set -euo pipefail
        cd ${REPO_DIR}
        $1
      "
}
