#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/evaluate_agent_benchmark.py \
  --public data/gerimedbench_asia76_public.json \
  --hidden data/gerimedbench_asia76_hidden_env.json \
  --pred outputs/ATLAS-Agent_agent_predictions.json \
  --summary results/summary_ATLAS-Agent_agent.recomputed.json \
  --detail results/detail_ATLAS-Agent_agent.recomputed.json

echo "[OK] recomputed ATLAS-Agent summary:"
cat results/summary_ATLAS-Agent_agent.recomputed.json
