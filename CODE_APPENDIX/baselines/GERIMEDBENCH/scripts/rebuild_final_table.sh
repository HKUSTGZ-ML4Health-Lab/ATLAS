#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/make_agent_table.py \
  --summaries \
    results/summary_ATLAS-Agent_agent.json \
    results/summary_Mistral-Small-3.2-24B-Instruct-2506_agent.json \
    results/summary_Qwen3-30B-A3B-Instruct-2507_agent.json \
    results/summary_DeepSeek-R1-Distill-Qwen-32B_agent.json \
    results/summary_MedGemma_27B_Text_agent.json \
    results/summary_Llama-3.3-70B-Instruct_agent.json \
  --out_md results/table3_gerimedbench_asia76_agent_final.recomputed.md \
  --out_csv results/table3_gerimedbench_asia76_agent_final.recomputed.csv

cat results/table3_gerimedbench_asia76_agent_final.recomputed.md
