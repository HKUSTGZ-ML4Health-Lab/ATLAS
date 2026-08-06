#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BUDGET="${BUDGET:-3}"
LIMIT="${LIMIT:-0}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_ROOT="${ATLAS_MODEL_ROOT:-$HOME/.cache/atlas_models}"

run_model() {
  echo "================================================================================"
  echo "[RUN] $METHOD"
  echo "================================================================================"
  METHOD="$METHOD" MODEL_DIR="$MODEL_DIR" SERVED="$SERVED" CUDA_DEVICES="$CUDA_DEVICES" GPU_UTIL="$GPU_UTIL" TP="$TP" LIMIT="$LIMIT" BUDGET="$BUDGET" MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_TOKENS="$MAX_TOKENS" VLLM_BIN="$VLLM_BIN" bash scripts/run_one_agent_llm.sh
}

METHOD="Qwen3-30B-A3B-Instruct-2507"; MODEL_DIR="${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507"; SERVED="qwen3_30b_a3b_instruct_2507"; CUDA_DEVICES="0"; GPU_UTIL="0.70"; TP="1"; MAX_MODEL_LEN="4096"; MAX_TOKENS="1024"; run_model
METHOD="DeepSeek-R1-Distill-Qwen-32B"; MODEL_DIR="${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-32B"; SERVED="deepseek_r1_distill_qwen32b"; CUDA_DEVICES="0"; GPU_UTIL="0.70"; TP="1"; MAX_MODEL_LEN="4096"; MAX_TOKENS="1024"; run_model
METHOD="MedGemma 27B Text"; MODEL_DIR="${MODEL_ROOT}/medgemma-27b-text-it"; SERVED="medgemma27b_text"; CUDA_DEVICES="0"; GPU_UTIL="0.68"; TP="1"; MAX_MODEL_LEN="4096"; MAX_TOKENS="1024"; run_model
METHOD="Llama-3.3-70B-Instruct"; MODEL_DIR="${MODEL_ROOT}/Llama-3.3-70B-Instruct"; SERVED="llama33_70b"; CUDA_DEVICES="0,1"; GPU_UTIL="0.78"; TP="2"; MAX_MODEL_LEN="4096"; MAX_TOKENS="1024"; run_model
