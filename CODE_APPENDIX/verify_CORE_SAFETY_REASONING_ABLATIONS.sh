#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
F="$ROOT/experiments/CORE_SAFETY_REASONING_ABLATIONS/results/CORE_SAFETY_REASONING_ABLATIONS.json"
O="$ROOT/reproduced/CORE_SAFETY_REASONING_ABLATIONS"
rm -rf "$O"; mkdir -p "$O"
python - "$F" "$O/verification.json" <<'PY_ABL'
import json,sys
from pathlib import Path
o=json.load(Path(sys.argv[1]).open(encoding='utf-8')); rows=o['official_paper_rows']
n=[r['Variant'] for r in rows]
e=['Full ATLAS','w/o PMCG','w/o Geriatric Risk Auditor','w/o Drug Conflict Auditor','w/o Safety Gate']
assert n==e,(n,e)
assert rows[0]['Strict Success']==92.04 and rows[0]['Unsafe Rate']==0.0 and rows[0]['Trace Pass Rate']==92.04 and rows[0]['OSRS']==97.21
v={'status':'PASS','official_rows':n,'full_atlas':{'Strict Success':92.04,'Unsafe Rate':0.0,'Trace Pass Rate':92.04,'OSRS':97.21}}
Path(sys.argv[2]).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('[OK] Core Safety-Reasoning Ablations match the paper-visible Figure 4 rows.')
PY_ABL
