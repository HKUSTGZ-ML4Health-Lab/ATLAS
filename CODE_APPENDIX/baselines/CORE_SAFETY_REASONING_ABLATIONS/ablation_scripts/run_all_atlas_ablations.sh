#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATLAS_ABLATION_ROOT="${ATLAS_ABLATION_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ATLAS_ABLATION_ROOT"

mkdir -p baselines_v2/logs baselines_v2/outputs baselines_v2/results

echo "[ABLATION START] $(date)"
echo "[NOTE] This runs only ATLAS ablations, not LLM baselines."

variants=(
  no_pmcg
  no_drug_conflict_auditor
  no_geriatric_risk_auditor
  no_safety_gate
  no_alternative_agent
  no_safety_critic_revision
  no_open_candidate_generation
  no_trace_verifier
)

for ab in "${variants[@]}"; do
  name="ablation_${ab}"

  echo
  echo "============================================================"
  echo "[RUN] $name"
  echo "============================================================"

  rm -f "baselines_v2/outputs/final_predictions_${name}.json"
  rm -f "baselines_v2/outputs/failures_${name}.json"
  rm -f "baselines_v2/results/eval_${name}.json"
  rm -f "baselines_v2/results/summary_${name}.json"

  PYTHONUNBUFFERED=1 python -u baselines_v2/runners/run_atlas_ablations.py \
    --ablation "$ab" \
    --name "$name" \
    | tee "baselines_v2/logs/run_${name}.log"

  python baselines_v2/runners/evaluate_one.py \
    --name "$name" \
    --pred "baselines_v2/outputs/final_predictions_${name}.json" \
    | tee "baselines_v2/logs/eval_${name}.log"
done

python baselines_v2/runners/make_table2_ablations.py \
  | tee baselines_v2/logs/make_table2_ablations.log

echo "[ABLATION DONE] $(date)"
