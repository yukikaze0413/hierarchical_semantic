#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
run_logged "check_data" python -m hsb_eeg2text.preprocessing.zuco --config "${CONFIG}" --audit-only --output-report outputs/reports/zuco_audit.json
