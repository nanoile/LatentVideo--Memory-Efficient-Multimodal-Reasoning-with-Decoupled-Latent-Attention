#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=init_conda_env.sh
source "$SCRIPT_DIR/init_conda_env.sh"
conda_init_and_activate
# shellcheck source=env_large_disk_caches.sh
source "$SCRIPT_DIR/env_large_disk_caches.sh"
cd "$REPO_ROOT"
# Ensure launcher-spawned workers inherit large-disk temp (avoids /tmp ENOSPC).
export TMPDIR TMP TEMP
# Propagate all cache vars sourced above (HF, ModelScope, pip, XDG) to child processes.
export XDG_CACHE_HOME HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE MODELSCOPE_CACHE PIP_CACHE_DIR
export TRITON_CACHE_DIR TORCH_EXTENSIONS_DIR REPO_ROOT

if command -v df >/dev/null 2>&1; then
  _root_kb=$(df -Pk / 2>/dev/null | awk "NR==2 {print \$4}")
  if [ -n "${_root_kb:-}" ] && [ "${_root_kb:-0}" -lt 1048576 ]; then
    echo "WARNING: root (/) has < 1GiB free (${_root_kb} KiB). Free space on / or training may SIGKILL/OOM; caches are redirected under REPO_ROOT."
    echo "         ZeRO-3 + CPU optimizer offload is host-RAM heavy with 4 ranks; try NUM_GPUS=2 (uses configs/deepspeed_config_cached_2gpu.json)."
  fi
fi

echo "=== Phase 2: Train Student with Cached Features ==="
echo "    NUM_GPUS=${NUM_GPUS}  DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
# Prerequisite: Phase 1 must populate cached_features/ with sample_000000.pt …
#   bash scripts/run_extract_features.sh
# (This script does not call the teacher model; it only loads those .pt files.)


# If the job dies with exit -9 right after "Epoch 1", check `free -h`: when swap is 100% full and / is
# out of space, the OOM killer often SIGKILLs a rank during the first ZeRO-3 optimizer step (not an NCCL bug).
# Fix: free space on /, add swap on a large disk (e.g. under /mnt/data), and/or use fewer GPUs (NUM_GPUS=2).

# NCCL optimization for 3090 without NVLink
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200
# Prefer torch's name (NCCL_ASYNC_ERROR_HANDLING is deprecated in newer NCCL bindings).
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# JIT compile: single arch + one job per process (large RAM spike otherwise). Ampere=8.6, Ada=8.9, Hopper=9.0.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-1}"
# Limit BLAS/OpenMP threads per process (4 ranks × many threads = huge RAM).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# Cuts glibc per-process arena bloat (4 ranks × arenas ≈ multi‑GB RSS spikes on optimizer step).
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

# ZeRO-3: params stay on GPU (sharded); only optimizer states CPU-offload — lowers RAM vs offload_param×ranks.
# NUM_GPUS: pick a config whose train_batch_size equals
#   train_micro_batch_size_per_gpu * NUM_GPUS * gradient_accumulation_steps (see JSON comments below).
NUM_GPUS="${NUM_GPUS:-4}"
if [ -n "${DEEPSPEED_CONFIG:-}" ]; then
  :
elif [ "$NUM_GPUS" = "1" ]; then
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG_1GPU:-configs/deepspeed_config_cached_1gpu.json}"
elif [ "$NUM_GPUS" = "2" ]; then
  DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG_2GPU:-configs/deepspeed_config_cached_2gpu.json}"
else
  DEEPSPEED_CONFIG="configs/deepspeed_config_cached.json"
fi
# numactl --interleave=all can raise RAM pressure on some boxes; opt in with USE_NUMACTL=1.
NUMAWRAP=()
if [ "${USE_NUMACTL:-0}" = 1 ] && command -v numactl >/dev/null 2>&1; then
  NUMAWRAP=(numactl --interleave=all)
fi

# Optional: EXTRA_PY_ARGS="--max_steps 20" for a quick multi-step smoke test.
# shellcheck disable=SC2086
"${NUMAWRAP[@]}" deepspeed --num_gpus="${NUM_GPUS}" experiments/phase3_distillation_cached.py \
    --deepspeed_config "${DEEPSPEED_CONFIG}" \
    --model_config configs/model_config.yaml \
    --distill_config configs/distill_config.yaml \
    --feature_dir cached_features \
    --output_dir checkpoints/v_mla \
    ${EXTRA_PY_ARGS-}

echo "✓ Training complete!"
