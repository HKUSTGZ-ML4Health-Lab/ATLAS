#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from framework_src.agents.unified_orchestrator import UnifiedOrchestratorAgent
from framework_src.system_impl.agents_impl import _append_message, _send, _record_yesno
from framework_src.system_impl.normalization import norm

class LegacySafetyCritic:
    name='SafetyCritic_NO_CANDIDATE_HARD_VETO'
    def run(self,state,rules=None):
        d=state.get('decision') or {}; rec={norm(x) for x in d.get('M_rec',[])}; avoid={norm(x) for x in d.get('M_avoid',[])}; caution={norm(x) for x in d.get('M_caution',[])}; alt={norm(x) for x in d.get('M_alt',[])}; board=state.get('yes_no_decision_board',{}); findings=[]
        if rec & avoid: findings.append({'type':'unsafe_overlap','severity':'critical','detail':sorted(rec&avoid),'resolved':False})
        if avoid and not alt: findings.append({'type':'missing_safer_alternative','severity':'moderate','detail':'M_avoid is non-empty but M_alt is empty','resolved':False})
        for med in d.get('M_rec',[]):
            if board.get(med,{}).get('should_avoid'): findings.append({'type':'recommendation_conflicts_with_yes_no_avoid_gate','severity':'moderate','detail':med,'resolved':False})
        for med in d.get('M_rec',[]):
            if norm(med) in caution and str((d.get('M_level') or {}).get(med,'')).upper()!='C': findings.append({'type':'caution_recommendation_without_c_level','severity':'moderate','detail':med,'resolved':False})
        trace=state.get('trace_verification') or {}
        if trace.get('unsupported_claims'): findings.append({'type':'unsupported_claims','severity':'moderate','detail':trace.get('unsupported_claims'),'resolved':False})
        state['critic_findings']=findings; state.setdefault('blackboard',{})['critic_findings']=findings
        _record_yesno(state,self.name,'Does the draft decision pass legacy safety critique?',not any(f.get('severity')=='critical' for f in findings),{'finding_count':len(findings)},target='final_decision',rationale='Direct ablation removes candidate hard veto.')
        _append_message(state,self.name,'pass' if not findings else 'review_required','Candidate hard veto disabled for direct ablation.',{'finding_count':len(findings)})
        return state

class LegacyRevisionAgent:
    name='RevisionAgent_NO_SLOT_PRESERVING_HARD_VETO_REPAIR'
    def run(self,state,rules=None):
        d=state.get('decision') or {}; rec=list(d.get('M_rec',[])); avoid=list(d.get('M_avoid',[])); changed=False; avoid_norm={norm(x) for x in avoid}; new_rec=[x for x in rec if norm(x) not in avoid_norm]
        if len(new_rec)!=len(rec): d['M_rec']=new_rec; changed=True
        alt=d.get('M_alt',[])
        if avoid and not alt:
            proposed=state.get('alternative_selection',{}).get('safer_alternative')
            if proposed and norm(proposed) not in avoid_norm: d['M_alt']=[proposed]; changed=True
        if any(f.get('severity')=='critical' and not f.get('resolved') for f in state.get('critic_findings',[])):
            d['U']=True; d['uncertainty_reason']='legacy safety critic requested clinician review'; changed=True
        if not d.get('M_rec'):
            proposed=state.get('alternative_selection',{}).get('safer_alternative')
            if proposed and norm(proposed) not in avoid_norm: d['M_rec']=[proposed]; d.setdefault('M_alt',[proposed]); changed=True
        state['decision']=d; state.setdefault('blackboard',{})['revised_decision']=d
        for f in state.get('critic_findings',[]):
            if f.get('type') in ('unsafe_overlap','missing_safer_alternative'): f['resolved']=True
        _append_message(state,self.name,'pass','Candidate hard-veto repair disabled for direct ablation.',{'changed':changed})
        return state

class DirectAblationOrchestrator(UnifiedOrchestratorAgent):
    def __init__(self,rules):
        super().__init__(rules)
        self.registry['safety_critique']=LegacySafetyCritic()
        self.registry['decision_revision']=LegacyRevisionAgent()

cases=json.load(open(ROOT/'02_FROZEN_INFERENCE/data/final_test_input.json',encoding='utf-8'))
rules=json.load(open(ROOT/'02_FROZEN_INFERENCE/frozen/frozen_rules.json',encoding='utf-8'))
orch=DirectAblationOrchestrator(rules)
preds=[orch.run_case(c) for c in cases]
out=ROOT/'ATLAS_ABLATION/baselines_v2/outputs/final_predictions_ablation_no_candidate_safety_gate_v2.json'
json.dump(preds,open(out,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(f'Wrote {len(preds)} predictions to {out}')
