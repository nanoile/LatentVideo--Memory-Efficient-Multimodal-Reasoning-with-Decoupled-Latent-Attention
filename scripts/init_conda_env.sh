#!/usr/bin/env bash
# Portable conda activation for scripts. Override env name: CONDA_ENV_NAME=myenv bash scripts/...
conda_init_and_activate() {
  local env_name="${1:-${CONDA_ENV_NAME:-zys_q}}"
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC2312
    eval "$(conda shell.bash hook 2>/dev/null)" && conda activate "$env_name" && return 0
  fi
  for _root in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    if [ -f "$_root/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "$_root/etc/profile.d/conda.sh"
      conda activate "$env_name"
      return 0
    fi
  done
  echo "ERROR: conda not found. Install Miniconda/Anaconda, or run: conda activate $env_name" >&2
  return 1
}
