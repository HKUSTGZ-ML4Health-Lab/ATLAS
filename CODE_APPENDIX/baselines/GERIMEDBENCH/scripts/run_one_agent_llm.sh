#!/usr/bin/env bash
set -euo pipefail
METHOD="${METHOD:?METHOD required}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR required}"
SERVED="${SERVED:?SERVED required}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
GPU_UTIL="${GPU_UTIL:-0.70}"
TP="${TP:-1}"
LIMIT="${LIMIT:-3}"
BUDGET="${BUDGET:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
VLLM_BIN="${VLLM_BIN:-vllm}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
PUB="data/gerimedbench_asia76_public.json"
HID="data/gerimedbench_asia76_hidden_env.json"
SAFE_NAME=$(echo "$METHOD" | tr ' /+' '___' | tr -cd 'A-Za-z0-9_.-')
PRED="outputs/${SAFE_NAME}_agent_predictions.json"
SUM="results/summary_${SAFE_NAME}_agent.json"
DET="results/detail_${SAFE_NAME}_agent.json"
LOG="logs/vllm_${SAFE_NAME}.log"
mkdir -p outputs results logs
cleanup() {
  pkill -9 -f "vllm" 2>/dev/null || true
  pkill -9 -f "api_server" 2>/dev/null || true
  pkill -9 -f "EngineCore" 2>/dev/null || true
  pkill -9 -f "ray::" 2>/dev/null || true
  pkill -9 -f "raylet" 2>/dev/null || true
  pkill -9 -f "gcs_server" 2>/dev/null || true
  pkill -9 -f "plasma_store" 2>/dev/null || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do kill -9 "$pid" 2>/dev/null || true; done
  rm -rf /dev/shm/ray* /dev/shm/vllm* /tmp/ray 2>/dev/null || true
  sleep 10
}
cleanup
nvidia-smi || true
EXTRA=()
if [[ "$TP" != "1" ]]; then EXTRA+=(--tensor-parallel-size "$TP"); fi
CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 VLLM_DISABLE_FLASHINFER=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 \
"$VLLM_BIN" serve "$MODEL_DIR" --served-model-name "$SERVED" --host 0.0.0.0 --port 8000 --trust-remote-code --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" --dtype bfloat16 --enforce-eager "${EXTRA[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 240); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[ERROR] vLLM server exited"; tail -120 "$LOG" || true; exit 1; fi
  if curl -s "${BASE_URL}/models" >/dev/null 2>&1; then echo "[OK] server ready"; break; fi
  if [[ "$i" == "240" ]]; then echo "[ERROR] server not ready"; tail -120 "$LOG" || true; exit 1; fi
  sleep 2
done
python scripts/run_agent_benchmark.py --mode llm --method "$METHOD" --model "$SERVED" --base_url "$BASE_URL" --api_key "$API_KEY" --public "$PUB" --hidden "$HID" --out "$PRED" --budget "$BUDGET" --limit "$LIMIT" --temperature 0.0 --max_tokens "$MAX_TOKENS"
python scripts/evaluate_agent_benchmark.py --public "$PUB" --hidden "$HID" --pred "$PRED" --summary "$SUM" --detail "$DET"
cat "$SUM"
cleanup
