#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=init_conda_env.sh
source "$SCRIPT_DIR/init_conda_env.sh"
conda_init_and_activate
source "$SCRIPT_DIR/env_large_disk_caches.sh"
cd "$REPO_ROOT"

echo "=== Phase 4: Evaluation ==="
python experiments/phase4_evaluation.py \
    --checkpoint checkpoints/v_mla/final \
    --benchmark mvbench \
    --output_dir results/evaluation
