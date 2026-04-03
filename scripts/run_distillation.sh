#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=init_conda_env.sh
source "$SCRIPT_DIR/init_conda_env.sh"
conda_init_and_activate
source "$SCRIPT_DIR/env_large_disk_caches.sh"
cd "$REPO_ROOT"

echo "=== Phase 3: Structural Distillation (Optimized) ==="

# NCCL optimization for 3090 without NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Increase NCCL timeout
export NCCL_TIMEOUT=7200

echo "Environment variables set:"
echo "  NCCL_P2P_DISABLE=1"
echo "  NCCL_IB_DISABLE=1"
echo "  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  NCCL_TIMEOUT=7200"

deepspeed --num_gpus=4 experiments/phase3_distillation.py \
    --deepspeed_config configs/deepspeed_config.json \
    --model_config configs/model_config.yaml \
    --distill_config configs/distill_config.yaml \
    --output_dir checkpoints/v_mla
