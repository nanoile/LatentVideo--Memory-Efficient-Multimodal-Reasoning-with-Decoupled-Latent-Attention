#!/usr/bin/env bash
# Triton autotune + PyTorch extension JIT: keep caches on the repo disk (e.g. /mnt/data/...) instead of a full $HOME.
# Usage: source "$(dirname "$0")/env_large_disk_caches.sh"   (from a script in scripts/)
_scripts_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$(cd "$_scripts_dir/.." && pwd)"
export TRITON_CACHE_DIR="$REPO_ROOT/.triton_cache"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.torch_extensions"
# gcc/cc1 write *.s under $TMPDIR; default /tmp often lives on a small root/home partition and fills up.
export TMPDIR="$REPO_ROOT/.tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
# HF / hub / pip: keep writable caches off a full root $HOME/.cache (avoids ENOSPC during compile or hub locks).
export XDG_CACHE_HOME="$REPO_ROOT/.cache/xdg"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export MODELSCOPE_CACHE="$REPO_ROOT/.cache/modelscope"
export PIP_CACHE_DIR="$REPO_ROOT/.cache/pip"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" \
  "$XDG_CACHE_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
  "$MODELSCOPE_CACHE" "$PIP_CACHE_DIR"
# Avoid multi-arch nvcc JIT (huge RAM) when DeepSpeed/torch compile extensions.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
