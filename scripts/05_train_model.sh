#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
read -r -p "Experiment name [zuco_mvp_deep4]: " exp
exp="${exp:-zuco_mvp_deep4}"
read -r -p "Epochs (blank for config default): " epochs
read -r -p "Smoke mode with hash text encoder? [y/N]: " smoke
echo "Training variant:"
echo "1) hierarchical"
echo "2) fine_only"
echo "3) no_hierarchy_loss"
echo "4) no_curriculum"
read -r -p "Choose variant [1]: " variant_choice
case "${variant_choice:-1}" in
  1) variant="hierarchical";;
  2) variant="fine_only";;
  3) variant="no_hierarchy_loss";;
  4) variant="no_curriculum";;
  *) echo "Invalid variant."; exit 2;;
esac
read -r -p "Shuffle training labels control? [y/N]: " shuffle
args=(python -m hsb_eeg2text.training.train --config "${CONFIG}" --experiment-name "${exp}")
if [[ -n "${epochs}" ]]; then
  args+=(--epochs "${epochs}")
fi
if [[ "${smoke}" == "y" || "${smoke}" == "Y" ]]; then
  args+=(--smoke)
fi
args+=(--variant "${variant}")
if [[ "${shuffle}" == "y" || "${shuffle}" == "Y" ]]; then
  args+=(--shuffle-labels)
fi
run_tmux_or_foreground "train_model" "${args[@]}"
