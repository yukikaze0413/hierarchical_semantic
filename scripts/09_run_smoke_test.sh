#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
read -r -p "Mock sentence count [24]: " sentences
sentences="${sentences:-24}"
run_logged "smoke_test" python scripts/09_run_smoke_test.py --config "${CONFIG}" --sentences "${sentences}"
