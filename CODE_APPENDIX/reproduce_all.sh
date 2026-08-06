#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT/reproduce_POLICY_DISTILLATION_STAGE1.sh"
bash "$ROOT/reproduce_WESTERN_MULTIMORBIDITY_EVALUATION_SET.sh"
bash "$ROOT/reproduce_GERIMEDBENCH.sh"
bash "$ROOT/reproduce_CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET.sh"
bash "$ROOT/verify_CORE_SAFETY_REASONING_ABLATIONS.sh"
bash "$ROOT/reproduce_BLINDED_CLINICIAN_EVALUATION.sh"
echo '[OK] Code-appendix reproduction checks passed.'
