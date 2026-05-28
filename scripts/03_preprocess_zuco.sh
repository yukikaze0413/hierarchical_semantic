#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
echo "Preprocessing modes:"
echo "1) mock smoke data"
echo "2) manifest CSV"
echo "3) ZuCo .mat frequency features"
read -r -p "Choose mode [1/2/3]: " mode
read -r -p "Sample/sentence limit (blank for default): " limit
limit_arg=()
if [[ -n "${limit}" ]]; then
  limit_arg=(--sample-limit "${limit}")
fi
if [[ "${mode}" == "1" ]]; then
  run_logged "preprocess_mock" python -m hsb_eeg2text.preprocessing.zuco --config "${CONFIG}" --mock "${limit_arg[@]}"
elif [[ "${mode}" == "2" ]]; then
  read -r -p "Manifest CSV path: " manifest
  run_logged "preprocess_manifest" python -m hsb_eeg2text.preprocessing.zuco --config "${CONFIG}" --manifest "${manifest}" "${limit_arg[@]}"
elif [[ "${mode}" == "3" ]]; then
  run_logged "preprocess_zuco_mat" python -m hsb_eeg2text.preprocessing.zuco_mat --config "${CONFIG}" "${limit_arg[@]}"
  echo "Check structure report: ${PROJECT_ROOT}/outputs/reports/zuco_mat_structure.json"
else
  echo "Invalid mode."
  exit 2
fi
