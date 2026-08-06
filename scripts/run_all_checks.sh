#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONHASHSEED=0

clean_bytecode() {
  find "$ROOT" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  find "$ROOT" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true
}

clean_bytecode
python "$ROOT/scripts/verify_code_appendix.py"
python "$ROOT/scripts/verify_gold_isolation.py"
python "$ROOT/scripts/verify_anonymity.py" --root "$ROOT" --strict
python "$ROOT/scripts/verify_manifests.py" --root "$ROOT"
while IFS= read -r -d '' script; do bash -n "$script"; done < <(find "$ROOT" -type f -name '*.sh' -print0)
bash "$ROOT/reproduce_all.sh"
clean_bytecode
printf '%s
' '[OK] Structural, isolation, anonymity, manifest, and result checks passed.'
