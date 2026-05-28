#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
read -r -p "Decoded JSONL (blank to skip): " decoded
read -r -p "Reconstructed JSONL (blank to skip): " reconstructed
read -r -p "Output metrics [outputs/reports/metrics.json]: " output
output="${output:-outputs/reports/metrics.json}"
args=(python -m hsb_eeg2text.evaluation.evaluate --config "${CONFIG}" --output "${output}")
if [[ -n "${decoded}" ]]; then
  args+=(--decoded "${decoded}")
fi
if [[ -n "${reconstructed}" ]]; then
  args+=(--reconstructed "${reconstructed}")
fi
run_logged "evaluate" "${args[@]}"
