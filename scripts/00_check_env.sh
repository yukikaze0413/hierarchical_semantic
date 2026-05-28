#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
run_logged "check_env" python -m hsb_eeg2text.utils.env --config "${CONFIG}" --output outputs/reports/env_audit.json
