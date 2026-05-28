#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
read -r -p "Vocabulary size [100]: " vocab
vocab="${vocab:-100}"
read -r -p "Random hierarchy control? [y/N]: " random
args=(python -m hsb_eeg2text.taxonomy.build --config "${CONFIG}" --vocab-size "${vocab}")
if [[ "${random}" == "y" || "${random}" == "Y" ]]; then
  args+=(--random-hierarchy)
fi
run_logged "build_taxonomy" "${args[@]}"
