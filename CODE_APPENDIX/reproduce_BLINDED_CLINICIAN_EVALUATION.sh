#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$ROOT/experiments/BLINDED_CLINICIAN_EVALUATION"
OUT="$ROOT/reproduced/BLINDED_CLINICIAN_EVALUATION"
mkdir -p "$OUT"

python "$EXP/scripts/verify_experiment_preservation.py" \
  --appendix-root "$ROOT" \
  --guard "$EXP/preservation_guard.json"

python "$EXP/scripts/validate_reported_summary.py" \
  --summary "$EXP/reported_results/reported_summary.json" \
  --source-docx "$EXP/reported_results/source/ATLAS_Blinded_Clinician_Evaluation_Reported_Summary.docx" \
  --out "$OUT/summary.json"
python "$EXP/scripts/render_reported_outputs.py" \
  --summary "$EXP/reported_results/reported_summary.json" \
  --out-dir "$OUT"

RATINGS="$EXP/private_inputs/ratings"
KEY="$EXP/private_inputs/PRIVATE_randomization_key.csv"
if compgen -G "$RATINGS/*.csv" >/dev/null && [[ -f "$KEY" ]]; then
  RAW_OUT="$OUT/raw_rating_aggregation"
  python "$EXP/scripts/aggregate_clinician_review.py" \
    --ratings-dir "$RATINGS" \
    --key "$KEY" \
    --out-dir "$RAW_OUT" \
    --bootstrap 10000 \
    --seed 20260729
  python "$EXP/scripts/compare_aggregate_to_reported.py" \
    --aggregated "$RAW_OUT/expert_review_results.json" \
    --reported "$EXP/reported_results/reported_summary.json"
  printf '%s\n' '[OK] Reviewer-level clinician evaluation aggregated and verified.'
else
  :
fi
