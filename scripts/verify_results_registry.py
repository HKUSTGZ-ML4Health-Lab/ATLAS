#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

TOL=0.011

def load(path):
    return json.load(Path(path).open(encoding='utf-8'))

def close(a,b):
    return abs(float(a)-float(b)) < TOL

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default=None)
    p.add_argument('--require-reproduced',action='store_true')
    a=p.parse_args()
    root=Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[1]
    r=load(root/'RESULT_REGISTRY.json')
    assert r['system']=='ATLAS'
    w=r['WESTERN_MULTIMORBIDITY_EVALUATION_SET']['metrics']
    assert close(w['Strict Success'],92.04) and close(w['Unsafe Rate'],0) and close(w['OSRS'],97.21)
    g=r['GERIMEDBENCH']['metrics']
    assert close(g['Final Strict'],23.68) and close(g['Agent OSRS'],64.09)
    s=r['CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET']['metrics']
    assert close(s['Strict Success'],94.12) and close(s['OSRS'],97.55)
    expected=['Full ATLAS','w/o PMCG','w/o Geriatric Risk Auditor','w/o Drug Conflict Auditor','w/o Safety Gate']
    assert r['CORE_SAFETY_REASONING_ABLATIONS']['official_rows']==expected
    c=r['BLINDED_CLINICIAN_EVALUATION']
    assert c['N']==40 and c['reviewers']==3
    cm=c['metrics']
    assert close(cm['ATLAS']['Correct.'],4.11) and close(cm['ATLAS']['Safety'],4.35)
    assert cm['ATLAS']['Unsafe cases']=='1/40' and cm['ATLAS']['Preferred cases']=='28/40'
    assert close(cm['Gemini 3.1 Pro Preview']['Correct.'],3.77)
    assert cm['Gemini 3.1 Pro Preview']['Unsafe cases']=='2/40' and cm['Tie cases']=='7/40'

    pbl=r['PROPRIETARY_LLM_BASELINES']
    assert pbl['shared_settings']['temperature']==0.0
    assert pbl['shared_settings']['inference_accessed_gold'] is False
    assert pbl['GERIMEDBENCH_protocol']['api_calls_per_model']==152
    models=pbl['models']
    assert models['GPT-5']['model_id']=='gpt-5-2025-08-07'
    assert models['Claude Opus 4.6']['model_id']=='claude-opus-4-6'
    assert models['Gemini 3.1 Pro Preview']['model_id']=='gemini-3.1-pro-preview'
    for model in models.values():
        assert model['results']['WESTERN_MULTIMORBIDITY_EVALUATION_SET']['status']=='Complete'
        assert model['results']['GERIMEDBENCH']['status']=='Complete'
        assert model['results']['CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET']['status']=='Complete'
    assert close(models['GPT-5']['results']['WESTERN_MULTIMORBIDITY_EVALUATION_SET']['metrics']['OSRS'],71.37)
    assert close(models['Claude Opus 4.6']['results']['GERIMEDBENCH']['metrics']['Agent OSRS'],62.87)
    assert close(models['Gemini 3.1 Pro Preview']['results']['CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET']['metrics']['Strict Success'],92.60)
    assert close(models['Gemini 3.1 Pro Preview']['results']['CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET']['metrics']['OSRS'],97.84)

    reproduced=root/'CODE_APPENDIX/reproduced'
    files={
      'western':reproduced/'WESTERN_MULTIMORBIDITY_EVALUATION_SET/summary.json',
      'geri':reproduced/'GERIMEDBENCH/summary.json',
      'single':reproduced/'CROSS_REGIONAL_SINGLE_DISEASE_GENERALIZATION_SET/summary.json',
      'ablation':reproduced/'CORE_SAFETY_REASONING_ABLATIONS/verification.json',
      'clinician':reproduced/'BLINDED_CLINICIAN_EVALUATION/summary.json',
    }
    if a.require_reproduced:
        for path in files.values(): assert path.is_file(),path
    if files['western'].is_file():
        x=load(files['western']);
        for k,v in w.items(): assert close(x[k],v),(k,x[k],v)
    if files['geri'].is_file():
        x=load(files['geri']); aliases={'Info Gain':'Info_Gain','Query Efficiency':'Query_Efficiency','Revision Accuracy':'Revision_Accuracy','Final Strict':'Final_Strict','Unsafe Rate':'Unsafe_Rate','Trace Consistency':'Trace_Consistency','Agent OSRS':'Agent_OSRS'}
        for k,v in g.items(): assert close(x[aliases[k]],v),(k,x[aliases[k]],v)
    if files['single'].is_file():
        x=load(files['single']);
        for k,v in s.items(): assert close(x[k],v),(k,x[k],v)
    if files['ablation'].is_file():
        x=load(files['ablation']); assert x['official_rows']==expected and x['status']=='PASS'
    if files['clinician'].is_file():
        x=load(files['clinician']); assert x['status']=='PASS' and x['cases']==40 and x['reviewers']==3
    print('[OK] Standalone paper-aligned result registry verified.')
if __name__=='__main__': main()
