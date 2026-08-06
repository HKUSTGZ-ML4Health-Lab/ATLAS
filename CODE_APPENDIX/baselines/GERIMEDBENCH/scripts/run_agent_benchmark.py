#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM baseline runner for GeriMedBench-Asia-76 agent benchmark.

Use this for OpenAI-compatible local vLLM baselines. ATLAS main-system results
are produced by run_atlas_agent_adapter.py.
"""
import argparse, json, re, urllib.request
from pathlib import Path
from typing import Any, Dict, List


def load_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save_json(obj,p):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def extract_json(text: str) -> Dict[str, Any]:
    text=(text or '').strip()
    try: return json.loads(text)
    except Exception: pass
    m=re.search(r'\{.*\}', text, flags=re.S)
    if not m: return {}
    try: return json.loads(m.group(0))
    except Exception: return {}

def chat(base_url, api_key, model, messages, temperature=0.0, max_tokens=1024):
    url=base_url.rstrip('/')+'/chat/completions'
    payload={'model':model,'messages':messages,'temperature':temperature,'max_tokens':max_tokens}
    req=urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type':'application/json','Authorization':f'Bearer {api_key}'})
    with urllib.request.urlopen(req, timeout=300) as resp:
        obj=json.loads(resp.read().decode('utf-8'))
    return obj['choices'][0]['message']['content']

def as_list(x):
    if x is None: return []
    if isinstance(x, list): return [str(v) for v in x if str(v).strip()]
    if isinstance(x, str): return [x] if x.strip() else []
    return [str(x)]

def allowed_types(public_case):
    out=[]
    for q in public_case.get('allowed_queries',[]) or []:
        qt=q.get('query_type') if isinstance(q,dict) else q
        if qt: out.append(str(qt))
    return out

def sanitize_queries(queries, allowed, budget):
    allowed_set=set(allowed); out=[]
    for q in queries or []:
        if isinstance(q,dict): q=q.get('query_type') or q.get('query') or q.get('name')
        q=str(q).strip()
        if q in allowed_set and q not in out:
            out.append(q)
        if len(out)>=budget: break
    return out

class HiddenEnv:
    def __init__(self, hidden_cases): self.by_id={c.get('case_id'):c for c in hidden_cases}
    def query(self,cid,q): return (self.by_id.get(cid) or {}).get('hidden_facts',{}).get(q,'not available')

def plan_queries(public_case,args):
    allowed=allowed_types(public_case)
    budget=args.budget or public_case.get('query_budget',3)
    prompt=f"""
You are a geriatric medication safety agent.
You see only the public case. Some safety-critical facts are hidden.
Choose up to {budget} query_type values from allowed_queries that are most important for medication safety.
Use only allowed query_type values. Do not invent query types.

Public case:
{json.dumps(public_case,ensure_ascii=False,indent=2)}

Return valid JSON only:
{{"queries": ["query_type_1", "query_type_2"], "reasoning_path": ["brief reason"]}}
""".strip()
    content=chat(args.base_url,args.api_key,args.model,[{'role':'system','content':'Return valid JSON only.'},{'role':'user','content':prompt}],args.temperature,args.max_tokens)
    obj=extract_json(content)
    queries=sanitize_queries(obj.get('queries',[]), allowed, budget)
    if not queries: queries=allowed[:budget]
    return queries, content

def final_report(public_case, acquired, args):
    candidates=(public_case.get('visible_input') or {}).get('candidate_medications', [])
    prompt=f"""
You are a geriatric medication safety agent.
Use the public case and acquired facts to update patient state and generate the final medication safety report.
Use only candidate medication strings from candidate_medications for M_rec, M_avoid, M_caution, and M_alt.
Set U=true if a safety-critical medication risk requiring avoidance exists.
M_caution should contain candidate(s) that require monitoring; if no caution candidate is needed, use an empty list.

Candidate medications:
{json.dumps(candidates,ensure_ascii=False)}

Public case:
{json.dumps(public_case,ensure_ascii=False,indent=2)}

Acquired facts:
{json.dumps(acquired,ensure_ascii=False,indent=2)}

Return valid JSON only:
{{
  "updated_patient_state": ["..."],
  "final_decision": {{
    "M_rec": ["..."],
    "M_avoid": ["..."],
    "M_caution": ["..."],
    "M_alt": ["..."],
    "U": true
  }},
  "E": {{"acquired_facts": ["..."], "reasoning_path": ["..."]}}
}}
""".strip()
    content=chat(args.base_url,args.api_key,args.model,[{'role':'system','content':'Return valid JSON only.'},{'role':'user','content':prompt}],args.temperature,args.max_tokens)
    obj=extract_json(content)
    obj['_raw_final_response']=content
    return obj

def run_case(public_case, env, args):
    cid=public_case.get('case_id')
    queries, raw_q=plan_queries(public_case,args)
    acquired={q:env.query(cid,q) for q in queries}
    query_log=[{'query_type':q,'answer':acquired[q],'source':'hidden_env_via_explicit_query'} for q in queries]
    out=final_report(public_case,acquired,args)
    fd=out.get('final_decision') if isinstance(out.get('final_decision'),dict) else {'M_rec':[],'M_avoid':[],'M_caution':[],'M_alt':[],'U':False}
    e=out.get('E') if isinstance(out.get('E'),dict) else {}
    if 'acquired_facts' not in e: e['acquired_facts']=queries
    if 'reasoning_path' not in e: e['reasoning_path']=[f'Queried {q}: {acquired[q]}' for q in queries]
    e['gold_used_by_inference']=False
    return {
        'case_id':cid,'method':args.method,'queries':queries,'query_log':query_log,'acquired_facts':acquired,
        'updated_patient_state': as_list(out.get('updated_patient_state')),
        'final_decision': fd,
        'final_report': fd,
        'E': e,
        '_raw_query_response': raw_q,
        '_raw_final_response': out.get('_raw_final_response',''),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['llm'], required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--public', required=True)
    ap.add_argument('--hidden', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--model', required=True)
    ap.add_argument('--base_url', default='http://127.0.0.1:8000/v1')
    ap.add_argument('--api_key', default='EMPTY')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max_tokens', type=int, default=1024)
    args=ap.parse_args()
    public=load_json(args.public)
    hidden=load_json(args.hidden)
    if args.limit and args.limit>0: public=public[:args.limit]
    env=HiddenEnv(hidden)
    preds=[]
    for i,pc in enumerate(public,1):
        print(f'[{i}/{len(public)}] {args.method} {pc.get("case_id")}', flush=True)
        try: preds.append(run_case(pc,env,args))
        except Exception as e:
            preds.append({'case_id':pc.get('case_id'),'method':args.method,'error':repr(e),'queries':[],'query_log':[],'acquired_facts':{},'updated_patient_state':[], 'final_decision':{'M_rec':[],'M_avoid':[],'M_caution':[],'M_alt':[],'U':False}, 'final_report':{'M_rec':[],'M_avoid':[],'M_caution':[],'M_alt':[],'U':False}, 'E':{'reasoning_path':[f'Runner error: {repr(e)}'],'acquired_facts':[],'gold_used_by_inference':False}})
    result={'benchmark':'GeriMedBench-Asia-76-Agent','method':args.method,'mode':'llm','query_budget':args.budget,'gold_used_by_inference':False,'hidden_env_used_only_via_explicit_query':True,'predictions':preds}
    save_json(result,args.out)
    print(f'[OK] wrote {args.out}')
if __name__=='__main__': main()
