#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E="$ROOT/experiments/CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET"
O="$ROOT/reproduced/CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET"
rm -rf "$O"; mkdir -p "$O"
python "$E/scripts/evaluate_v3_official.py" \
 --pred "$E/outputs/ATLAS_predictions.json" \
 --gold "$E/data/reference.json" \
 --metrics "$E/evaluator_snapshot/framework_src/evaluation/metrics.py" \
 --out "$O/evaluation.json" --compact-out "$O/compact.json"
python - "$O/evaluation.json" "$O/summary.json" <<'PY_SINGLE'
import json,sys
from pathlib import Path
o=json.load(Path(sys.argv[1]).open(encoding='utf-8'));s=o['summary'];m=s['macro'];z=s['overall_safety_reasoning']
a={'N':s['total_cases'],'Strict Success':round(100*s['success_rate_strict'],2),'M_rec F1':round(100*m['M_rec_f1'],2),'M_avoid Recall':round(100*m['M_avoid_recall'],2),'M_avoid F1':round(100*m['M_avoid_f1'],2),'M_caution F1':round(100*m['M_caution_f1'],2),'M_alt F1':round(100*m['M_alt_f1'],2),'Unsafe Rate':round(100*s['unsafe_recommendation_rate'],2),'Trace Pass Rate':round(100*s['trace_consistency_pass_rate'],2),'OSRS':round(z['overall_safety_reasoning_score_percent'],2)}
e={'N':612,'Strict Success':94.12,'M_rec F1':91.83,'M_avoid Recall':98.04,'M_avoid F1':92.81,'M_caution F1':96.08,'M_alt F1':86.60,'Unsafe Rate':0.0,'Trace Pass Rate':100.0,'OSRS':97.55}
for k,v in e.items(): assert abs(float(a[k])-float(v))<0.011,(k,a[k],v)
Path(sys.argv[2]).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(a,ensure_ascii=False,indent=2))
print('[OK] Cross-Regional Single-Disease Generalization Set evaluation reproduced.')
PY_SINGLE
