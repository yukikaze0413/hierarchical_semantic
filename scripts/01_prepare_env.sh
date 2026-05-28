#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
show_context
echo "This will create or update the conda environment from environment.yml."
confirm "Prepare conda environment?"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first."
  exit 1
fi
if conda env list | awk '{print $1}' | grep -qx "hsb-eeg2text"; then
  (cd "${PROJECT_ROOT}" && conda env update -f environment.yml --prune)
else
  (cd "${PROJECT_ROOT}" && conda env create -f environment.yml)
fi
echo "Environment ready. Activate with: conda activate hsb-eeg2text"
