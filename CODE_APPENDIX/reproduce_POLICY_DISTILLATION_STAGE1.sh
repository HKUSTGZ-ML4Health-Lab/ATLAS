#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$ROOT/01_DEV39_WORKSPACE/policy_distillation"
export PYTHONHASHSEED=0
python "$PD/scripts/generate_teacher_trajectories.py"
python "$PD/scripts/extract_symbolic_policy.py"
python "$PD/scripts/verify_stage1_integrity.py"
echo '[OK] Reproduced all 117 Dev39 teacher trajectories and verified exact identity with the frozen runtime policy used by reported inference.'
