#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG="configs/zuco_mvp.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

confirm() {
  local prompt="${1:-Continue?}"
  read -r -p "${prompt} [y/N] " answer
  case "${answer}" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 1 ;;
  esac
}

show_context() {
  echo "Project root: ${PROJECT_ROOT}"
  echo "Config: ${CONFIG}"
  echo "Python: $(command -v python || true)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
  else
    echo "nvidia-smi not found"
  fi
}

run_logged() {
  local stage="$1"
  shift
  mkdir -p "${PROJECT_ROOT}/outputs/logs"
  local log="${PROJECT_ROOT}/outputs/logs/${stage}_$(timestamp).log"
  show_context
  echo "Log: ${log}"
  echo "Command: $*"
  confirm "Run ${stage}?"
  (cd "${PROJECT_ROOT}" && "$@" 2>&1 | tee "${log}")
}

run_tmux_or_foreground() {
  local stage="$1"
  shift
  mkdir -p "${PROJECT_ROOT}/outputs/logs"
  local log="${PROJECT_ROOT}/outputs/logs/${stage}_$(timestamp).log"
  local session="hsb_${stage}_$(timestamp)"
  local cmd="cd '${PROJECT_ROOT}' && export PYTHONPATH='${PROJECT_ROOT}/src:${PYTHONPATH:-}' && $* 2>&1 | tee '${log}'"
  show_context
  echo "Log: ${log}"
  echo "Command: $*"
  confirm "Run ${stage}?"
  if command -v tmux >/dev/null 2>&1; then
    tmux new-session -d -s "${session}" "${cmd}"
    echo "Started tmux session: ${session}"
    echo "Attach with: tmux attach -t ${session}"
  else
    echo "tmux not found; running in foreground."
    bash -lc "${cmd}"
  fi
}
