#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E="$ROOT/experiments/GERIMEDBENCH"
O="$ROOT/reproduced/GERIMEDBENCH"
rm -rf "$O"; mkdir -p "$O"
python "$E/scripts/evaluate_agent_benchmark.py" \
 --public "$E/data/public.json" \
 --hidden "$E/data/hidden_environment.json" \
 --pred "$E/outputs/ATLAS_predictions.json" \
 --summary "$O/summary.json" --detail "$O/detail.json"
python - "$O/summary.json" <<'PY_GERI'
import json,sys
from pathlib import Path
s=json.load(Path(sys.argv[1]).open(encoding='utf-8'))
e={'N':76,'Info_Gain':73.90,'Query_Efficiency':35.75,'Revision_Accuracy':44.17,'Final_Strict':23.68,'Unsafe_Rate':0.0,'Trace_Consistency':85.53,'Agent_OSRS':64.09}
for k,v in e.items(): assert abs(float(s[k])-float(v))<0.011,(k,s[k],v)
print(json.dumps(s,ensure_ascii=False,indent=2))
print('[OK] GeriMedBench evaluation completed with the locked evaluator.')
PY_GERI
