#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONHASHSEED=0

mkdir -p \
  01_DEV39_WORKSPACE/outputs \
  01_DEV39_WORKSPACE/results \
  02_FROZEN_INFERENCE/outputs \
  03_OFFLINE_EVALUATION/predictions \
  03_OFFLINE_EVALUATION/results

O="$ROOT/reproduced/WESTERN_MULTIMORBIDITY_EVALUATION_SET"
rm -rf "$O"
mkdir -p "$O"

rm -f 01_DEV39_WORKSPACE/outputs/* 01_DEV39_WORKSPACE/results/*
rm -f 02_FROZEN_INFERENCE/outputs/*
rm -f 03_OFFLINE_EVALUATION/predictions/* 03_OFFLINE_EVALUATION/results/*

python 01_DEV39_WORKSPACE/scripts/run_dev.py
python 01_DEV39_WORKSPACE/scripts/evaluate_dev.py
python 02_FROZEN_INFERENCE/scripts/run_final.py
cp 02_FROZEN_INFERENCE/outputs/final_predictions.json \
   03_OFFLINE_EVALUATION/predictions/final_predictions.json
python 03_OFFLINE_EVALUATION/scripts/evaluate_final.py

python - "$ROOT/03_OFFLINE_EVALUATION/results/final_evaluation.json" "$O/summary.json" <<'PY_WESTERN'
import json, sys
from pathlib import Path
source=Path(sys.argv[1]); target=Path(sys.argv[2])
o=json.load(source.open(encoding='utf-8'))
s=o['summary']; m=s['micro']; osrs=s['overall_safety_reasoning']
a={
'N':s['total_cases'], 'Strict Success':round(100*s['success_rate_strict'],2),
'M_rec F1':round(100*m['M_rec']['f1'],2), 'M_avoid Recall':round(100*m['M_avoid']['recall'],2),
'M_avoid F1':round(100*m['M_avoid']['f1'],2), 'M_caution F1':round(100*m['M_caution']['f1'],2),
'M_alt F1':round(100*m['M_alt']['f1'],2), 'Unsafe Rate':round(100*s['unsafe_recommendation_rate'],2),
'Trace Pass Rate':round(100*s['trace_consistency_pass_rate'],2),
'OSRS':round(osrs['overall_safety_reasoning_score_percent'],2)}
e={'N':201,'Strict Success':92.04,'M_rec F1':92.50,'M_avoid Recall':100.00,'M_avoid F1':91.57,'M_caution F1':75.47,'M_alt F1':92.84,'Unsafe Rate':0.00,'Trace Pass Rate':92.04,'OSRS':97.21}
for k,v in e.items(): assert abs(float(a[k])-float(v)) < 0.011,(k,a[k],v)
target.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(a,ensure_ascii=False,indent=2))
print('[OK] Western Multimorbidity Evaluation Set reproduced end to end.')
PY_WESTERN

cp 03_OFFLINE_EVALUATION/results/final_evaluation.json "$O/final_evaluation.json"
sha256sum \
  02_FROZEN_INFERENCE/outputs/final_predictions.json \
  03_OFFLINE_EVALUATION/results/final_evaluation.json \
  > "$O/artifacts.sha256"
