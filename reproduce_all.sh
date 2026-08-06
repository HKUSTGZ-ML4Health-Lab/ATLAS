#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONHASHSEED=0
bash "$ROOT/CODE_APPENDIX/reproduce_all.sh"
python "$ROOT/scripts/verify_results_registry.py" --require-reproduced
printf '%s
' '[OK] ATLAS paper-aligned experiments and evaluation checks passed.'
