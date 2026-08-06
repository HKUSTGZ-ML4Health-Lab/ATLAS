#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$ROOT/CODE_APPENDIX/reproduce_BLINDED_CLINICIAN_EVALUATION.sh"
