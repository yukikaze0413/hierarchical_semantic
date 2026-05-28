#!/usr/bin/env bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh" "$@"
echo "LLM backend:"
echo "1) DeepSeek V4 Pro"
echo "2) DeepSeek V4 Flash"
echo "3) Local Transformers Qwen"
echo "4) Custom OpenAI-compatible API"
echo "5) Mock backend"
read -r -p "Choose backend [1]: " choice
choice="${choice:-1}"
case "${choice}" in
  1) backend="deepseek_api";;
  2) backend="deepseek_flash";;
  3) backend="local_qwen";;
  4)
    backend="custom_openai"
    read -r -p "LLM_BASE_URL: " base_url
    read -r -p "LLM_MODEL: " model
    export LLM_BASE_URL="${base_url}"
    export LLM_MODEL="${model}"
    if [[ -z "${LLM_API_KEY:-}" ]]; then
      echo "Set LLM_API_KEY before using custom_openai."
      exit 1
    fi
    ;;
  5) backend="mock";;
  *) echo "Invalid backend."; exit 2;;
esac
if [[ "${backend}" == deepseek_* && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "Set DEEPSEEK_API_KEY before using DeepSeek API."
  exit 1
fi
read -r -p "Decoded JSONL [outputs/decoded_anchors/decoded_test.jsonl]: " decoded
decoded="${decoded:-outputs/decoded_anchors/decoded_test.jsonl}"
read -r -p "Output JSONL [outputs/reconstructed_sentences/reconstructed.jsonl]: " output
output="${output:-outputs/reconstructed_sentences/reconstructed.jsonl}"
echo "Reconstruction mode:"
echo "1) hierarchical_anchors"
echo "2) flat_keywords"
echo "3) oracle_anchors"
echo "4) no_rag"
read -r -p "Choose mode [1]: " mode_choice
case "${mode_choice:-1}" in
  1) mode="hierarchical_anchors";;
  2) mode="flat_keywords";;
  3) mode="oracle_anchors";;
  4) mode="no_rag";;
  *) echo "Invalid reconstruction mode."; exit 2;;
esac
run_tmux_or_foreground "reconstruct_sentences" python -m hsb_eeg2text.inference.reconstruct --config "${CONFIG}" --decoded "${decoded}" --backend "${backend}" --output "${output}" --reconstruction-mode "${mode}"
