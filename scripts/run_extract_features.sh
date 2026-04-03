#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=init_conda_env.sh
source "$SCRIPT_DIR/init_conda_env.sh"
conda_init_and_activate
source "$SCRIPT_DIR/env_large_disk_caches.sh"
cd "$REPO_ROOT"

echo "=== Phase 1: Extract Teacher Features ==="

python experiments/extract_teacher_features.py \
    --model_config configs/model_config.yaml \
    --distill_config configs/distill_config.yaml \
    --output_dir cached_features \
    --batch_size 4

echo "✓ Feature extraction complete!"
