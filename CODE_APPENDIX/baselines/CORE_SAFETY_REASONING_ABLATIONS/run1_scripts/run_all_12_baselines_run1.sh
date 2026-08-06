#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATLAS_ROOT="${ATLAS_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_ROOT="${ATLAS_MODEL_ROOT:-$HOME/.cache/atlas_models}"
cd "$ATLAS_ROOT"

mkdir -p baselines_v2/logs baselines_v2/model_logs baselines_v2/outputs baselines_v2/results baselines_v2/manifests

VLLM_BIN="${VLLM_BIN:-vllm}"
BASE_URL=http://127.0.0.1:8000/v1
API_KEY=EMPTY
MAX_TOKENS=2048
TEMP=0.0
TOP_K=8

echo "[RUN1 ALL 12 START] $(date)"
echo "[INFO] run1 naming rule: no suffix"
echo "[FAIRNESS] MAX_TOKENS=${MAX_TOKENS}, TEMP=${TEMP}, TOP_K=${TOP_K}, max_model_len=8192"

is_done() {
  local name="$1"
  local pred="baselines_v2/outputs/final_predictions_${name}.json"
  local summ="baselines_v2/results/summary_${name}.json"

  if [ -f "$pred" ] && [ -f "$summ" ]; then
    python - "$pred" <<'PY'
import json, sys
p = sys.argv[1]
try:
    data = json.load(open(p, encoding="utf-8"))
except Exception:
    sys.exit(1)
ids = {str(x.get("case_id")) for x in data}
if len(data) == 201 and len(ids) == 201:
    sys.exit(0)
sys.exit(1)
PY
    return $?
  fi

  return 1
}

kill_server() {
  echo "[CLEAN SERVER]"

  pkill -9 -f "vllm serve" 2>/dev/null || true
  pkill -9 -f "api_server" 2>/dev/null || true
  pkill -9 -f "ray::" 2>/dev/null || true
  pkill -9 -f "raylet" 2>/dev/null || true
  pkill -9 -f "gcs_server" 2>/dev/null || true
  pkill -9 -f "plasma_store" 2>/dev/null || true

  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    echo "[KILL GPU PID] $pid"
    kill -9 "$pid" 2>/dev/null || true
  done

  rm -rf /dev/shm/ray* 2>/dev/null || true
  rm -rf /dev/shm/vllm* 2>/dev/null || true
  rm -rf /tmp/ray 2>/dev/null || true

  sleep 15
}

wait_ready() {
  local served="$1"
  local pidfile="$2"
  local logfile="$3"

  for i in $(seq 1 180); do
    if curl -s http://127.0.0.1:8000/v1/models | grep -q "$served"; then
      echo "[OK] server ready: $served"
      curl -s http://127.0.0.1:8000/v1/models
      return 0
    fi

    if [ -f "$pidfile" ] && ! ps -p "$(cat "$pidfile")" >/dev/null 2>&1; then
      echo "[ERROR] server died: $served"
      tail -n 300 "$logfile" || true
      exit 1
    fi

    echo "[WAIT] $served not ready: $i/180"
    sleep 5
  done

  echo "[ERROR] server timeout: $served"
  tail -n 300 "$logfile" || true
  exit 1
}

start_server_single() {
  local model_dir="$1"
  local served="$2"
  local gpu_util="$3"

  if [ ! -e "$model_dir" ]; then
    echo "[ERROR] model dir missing: $model_dir"
    exit 1
  fi

  kill_server

  local logfile="baselines_v2/model_logs/${served}_server_run1.log"
  local pidfile="baselines_v2/model_logs/${served}_server_run1.pid"

  echo "[START SERVER SINGLE] $served"

  CUDA_VISIBLE_DEVICES=0 \
  TORCH_COMPILE_DISABLE=1 \
  TORCHINDUCTOR_DISABLE=1 \
  VLLM_DISABLE_FLASHINFER=1 \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_BIN" serve "$model_dir" \
    --served-model-name "$served" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization "$gpu_util" \
    --dtype bfloat16 \
    --enforce-eager \
    > "$logfile" 2>&1 &

  echo $! > "$pidfile"
  wait_ready "$served" "$pidfile" "$logfile"
}

start_server_tp2() {
  local model_dir="$1"
  local served="$2"
  local gpu_util="$3"

  if [ ! -e "$model_dir" ]; then
    echo "[ERROR] model dir missing: $model_dir"
    exit 1
  fi

  kill_server

  local logfile="baselines_v2/model_logs/${served}_server_run1.log"
  local pidfile="baselines_v2/model_logs/${served}_server_run1.pid"

  echo "[START SERVER TP2] $served"

  CUDA_VISIBLE_DEVICES=0,1 \
  TORCH_COMPILE_DISABLE=1 \
  TORCHINDUCTOR_DISABLE=1 \
  VLLM_DISABLE_FLASHINFER=1 \
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_BIN" serve "$model_dir" \
    --served-model-name "$served" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization "$gpu_util" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --enforce-eager \
    > "$logfile" 2>&1 &

  echo $! > "$pidfile"
  wait_ready "$served" "$pidfile" "$logfile"
}

run_eval() {
  local name="$1"
  local pred="baselines_v2/outputs/final_predictions_${name}.json"

  if [ ! -f "$pred" ]; then
    echo "[ERROR] prediction file missing: $pred"
    exit 1
  fi

  python baselines_v2/runners/evaluate_one.py \
    --name "$name" \
    --pred "$pred" \
    | tee "baselines_v2/logs/eval_${name}.log"
}

clean_artifacts() {
  local name="$1"
  rm -f "baselines_v2/outputs/final_predictions_${name}.json"
  rm -f "baselines_v2/outputs/failures_${name}.json"
  rm -f "baselines_v2/results/summary_${name}.json"
  rm -f "baselines_v2/results/eval_${name}.json"
}

run_cpu_first_batch() {
  local method="$1"

  if is_done "$method"; then
    echo "[SKIP DONE] $method"
    return 0
  fi

  echo "[RUN CPU BASELINE] $method"

  clean_artifacts "$method"

  PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_first_batch_baselines.py \
    --method "$method" \
    | tee "baselines_v2/logs/run_${method}.log"

  run_eval "$method"
}

run_pykeen() {
  local name="pykeen_rotate"

  if is_done "$name"; then
    echo "[SKIP DONE] $name"
    return 0
  fi

  echo "[RUN PYKEEN] $name"

  clean_artifacts "$name"

  if python baselines_v2/runners/run_pykeen_rotate.py --help 2>&1 | grep -q -- "--name"; then
    PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_pykeen_rotate.py \
      --name "$name" \
      | tee "baselines_v2/logs/run_${name}.log"
  else
    PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_pykeen_rotate.py \
      | tee "baselines_v2/logs/run_${name}.log"
  fi

  run_eval "$name"
}

run_atlas() {
  local name="atlas"

  if is_done "$name"; then
    echo "[SKIP DONE] $name"
    return 0
  fi

  echo "[RUN ATLAS] $name"

  clean_artifacts "$name"

  rm -f 02_FROZEN_INFERENCE/outputs/final_predictions.json
  rm -f 03_OFFLINE_EVALUATION/predictions/final_predictions.json
  rm -f 03_OFFLINE_EVALUATION/results/final_evaluation.json

  PYTHONUNBUFFERED=1 bash run_final_inference.sh \
    | tee "baselines_v2/logs/run_${name}.log"

  if [ ! -f 02_FROZEN_INFERENCE/outputs/final_predictions.json ]; then
    echo "[ERROR] ATLAS final_predictions.json not generated"
    exit 1
  fi

  cp 02_FROZEN_INFERENCE/outputs/final_predictions.json \
     "baselines_v2/outputs/final_predictions_${name}.json"

  echo "[]" > "baselines_v2/outputs/failures_${name}.json"

  run_eval "$name"
}

run_llm_eval() {
  local name="$1"
  local served="$2"

  if is_done "$name"; then
    echo "[SKIP DONE] $name"
    return 0
  fi

  echo "[RUN LLM] $name"

  clean_artifacts "$name"

  PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_openai_compatible_llm.py \
    --name "$name" \
    --model "$served" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    | tee "baselines_v2/logs/run_${name}.log"

  run_eval "$name"
}

run_bm25_eval() {
  local name="bm25_llama33_70b_rag"
  local served="llama33_70b"

  if is_done "$name"; then
    echo "[SKIP DONE] $name"
    return 0
  fi

  echo "[RUN BM25 RAG] $name"

  clean_artifacts "$name"

  PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_bm25_rag_openai.py \
    --name "$name" \
    --model "$served" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    | tee "baselines_v2/logs/run_${name}.log"

  run_eval "$name"
}

ensure_bge_index() {
  echo "[CHECK BGE FAISS INDEX]"

  if python - <<'PY'
from pathlib import Path
candidates = []
for pat in ["*.faiss", "*.index"]:
    candidates += list(Path("baselines_v2").rglob(pat))
raise SystemExit(0 if candidates else 1)
PY
  then
    echo "[OK] BGE/FAISS index exists"
  else
    echo "[BUILD] BGE-M3 FAISS index missing; building now"
    PYTHONUNBUFFERED=1 python -u baselines_v2/runners/build_bge_faiss_index.py \
      | tee baselines_v2/logs/build_bge_faiss_index_run1.log
  fi
}

run_bge_eval() {
  local name="bge_m3_faiss_medgemma27b_rag"
  local served="medgemma27b_text"

  if is_done "$name"; then
    echo "[SKIP DONE] $name"
    return 0
  fi

  echo "[RUN BGE FAISS RAG] $name"

  ensure_bge_index
  clean_artifacts "$name"

  PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_bge_faiss_rag_openai.py \
    --name "$name" \
    --model "$served" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    | tee "baselines_v2/logs/run_${name}.log"

  run_eval "$name"
}

run_mdagents_eval() {
  local name="mdagents_shared_rag"
  local served="medgemma27b_text"

  if is_done "$name"; then
    echo "[SKIP DONE] $name"
    return 0
  fi

  echo "[RUN MDAGENTS SHARED RAG] $name"

  clean_artifacts "$name"

  PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_mdagents_shared_rag_openai.py \
    --name "$name" \
    --model "$served" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --top-k "$TOP_K" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    | tee "baselines_v2/logs/run_${name}.log"

  run_eval "$name"
}

run_cpu_first_batch retrieval_only_engine
run_cpu_first_batch frozen_guideline
run_cpu_first_batch generic_kg
run_pykeen
run_atlas

start_server_single ${MODEL_ROOT}/Mistral-Small-3.2-24B-Instruct-2506 mistral_small_32_24b 0.84
run_llm_eval mistral_small_32_24b_llm_only mistral_small_32_24b

start_server_tp2 ${MODEL_ROOT}/medgemma-27b-text-it medgemma27b_text 0.80
run_llm_eval medgemma27b_text_llm_only medgemma27b_text

start_server_single ${MODEL_ROOT}/Qwen3-30B-A3B-Instruct-2507 qwen3_30b_a3b_instruct_2507 0.84
run_llm_eval qwen3_30b_a3b_instruct_2507_llm_only qwen3_30b_a3b_instruct_2507

start_server_single ${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-32B deepseek_r1_distill_qwen32b 0.84
run_llm_eval deepseek_r1_distill_qwen32b_llm_only deepseek_r1_distill_qwen32b

start_server_tp2 ${MODEL_ROOT}/Llama-3.3-70B-Instruct llama33_70b 0.82
run_bm25_eval

start_server_tp2 ${MODEL_ROOT}/medgemma-27b-text-it medgemma27b_text 0.80
run_bge_eval
run_mdagents_eval

kill_server

python baselines_v2/runners/make_table1_run1.py | tee baselines_v2/logs/make_table1_run1_after_run_all_12.log

echo "[RUN1 ALL 12 DONE] $(date)"
