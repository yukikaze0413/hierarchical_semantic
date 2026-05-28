#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
read -r -p "Checkpoint path [outputs/checkpoints/zuco_mvp_deep4/latest.pt]: " ckpt
ckpt="${ckpt:-outputs/checkpoints/zuco_mvp_deep4/latest.pt}"
read -r -p "Split [test]: " split
split="${split:-test}"
read -r -p "Output path [outputs/decoded_anchors/decoded_${split}.jsonl]: " output
output="${output:-outputs/decoded_anchors/decoded_${split}.jsonl}"
read -r -p "Smoke/hash text encoder? [y/N]: " smoke
args=(python -m hsb_eeg2text.inference.decode --config "${CONFIG}" --checkpoint "${ckpt}" --split "${split}" --output "${output}")
if [[ "${smoke}" == "y" || "${smoke}" == "Y" ]]; then
  args+=(--smoke)
fi
run_logged "decode_anchors" "${args[@]}"
