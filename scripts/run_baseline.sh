#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=init_conda_env.sh
source "$SCRIPT_DIR/init_conda_env.sh"
conda_init_and_activate
source "$SCRIPT_DIR/env_large_disk_caches.sh"
cd "$REPO_ROOT"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"

echo "=== Phase 1: Baseline Profiling ==="
python experiments/phase1_baseline.py \
    --model_name "$MODEL_NAME" \
    --video_lengths 60,120,300,600,1800 \
    --output_dir results/baseline
