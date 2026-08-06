
from __future__ import annotations
from .clinical_state_grounder import ClinicalStateGrounder
from .guideline_candidate_agent import GuidelineCandidateAgent
from .drug_conflict_auditor import DrugConflictAuditor
from .geriatric_risk_auditor import GeriatricRiskAuditor
from .alternative_agent import AlternativeAgent
from ..pmcg.builder import PMCGBuilder
from .risk_benefit_deliberator import RiskBenefitDeliberator
from .safety_critic import SafetyCritic
from .revision_agent import RevisionAgent
from .trace_verifier import TraceVerifier
from ..system_impl.normalization import norm

import inspect
class UnifiedOrchestratorAgent:
    name = 'UnifiedOrchestratorAgent'
    def __init__(self, rules, max_rounds=4):
        self.rules = rules
        self.max_rounds = max_rounds
        self.registry = {
            'clinical_state_grounding': ClinicalStateGrounder(),
            'candidate_generation': GuidelineCandidateAgent(),
            'drug_conflict_audit': DrugConflictAuditor(),
            'geriatric_risk_audit': GeriatricRiskAuditor(),
            'alternative_search': AlternativeAgent(),
            'pmcg_building': PMCGBuilder(),
            'risk_benefit_deliberation': RiskBenefitDeliberator(),
            'safety_critique': SafetyCritic(),
            'decision_revision': RevisionAgent(),
            'trace_verification': TraceVerifier(),
        }
    def _initialize_state(self, case):
        return {'case': case, 'rules': self.rules, 'messages': [], 'agent_interactions': [], 'blackboard': {}, 'orchestration_log': [], 'yes_no_judgments': [], 'completed_agents': [], 'critic_findings': [], 'iteration': 0}
    def _log(self, state, event, payload=None):
        state.setdefault('orchestration_log', []).append({'round': state.get('iteration', 0), 'event': event, 'payload': payload or {}})
    def _state_signature(self, state):
        pmcg = state.get('pmcg') or {'nodes': [], 'edges': []}
        decision = state.get('decision') or {}
        return {'candidate_pool': len(state.get('grounded', {}).get('candidate_pool', [])), 'pmcg_edges': len(pmcg.get('edges', [])), 'critic_findings': len(state.get('critic_findings', [])), 'M_rec': len(decision.get('M_rec', [])), 'M_avoid': len(decision.get('M_avoid', [])), 'M_caution': len(decision.get('M_caution', [])), 'M_alt': len(decision.get('M_alt', []))}
    def _run_task(self, state, task_name):
        agent = self.registry[task_name]
        before = self._state_signature(state)
        self._log(state, 'dispatch', {'task': task_name, 'agent': getattr(agent, 'name', task_name), 'before': before})
        method = 'refresh_after_audits' if task_name == 'pmcg_building' and state.get('pmcg') else 'run'
        fn = getattr(agent, method) if hasattr(agent, method) else agent.run

        # Runtime compatibility only:
        # Some agents implement run(state), others implement run(state, rules).
        # No clinical rule, PMKG, scoring, or gold access is changed.
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        has_varargs = any(x.kind == x.VAR_POSITIONAL for x in params)

        if has_varargs or len(params) >= 2:
            state = fn(state, self.rules)
        else:
            state = fn(state)
        after = self._state_signature(state)
        state.setdefault('completed_agents', []).append(task_name)
        self._log(state, 'state_changed' if before != after else 'state_unchanged', {'task': task_name, 'agent': getattr(agent, 'name', task_name), 'after': after})
        return state
    def _initial_tasks(self):
        return ['clinical_state_grounding','candidate_generation','pmcg_building','drug_conflict_audit','geriatric_risk_audit','alternative_search','risk_benefit_deliberation','safety_critique','decision_revision','trace_verification']
    def _needs_review(self, state):
        tasks=[]
        decision = state.get('decision') or {}
        rec = {norm(x) for x in decision.get('M_rec', [])}
        avoid = {norm(x) for x in decision.get('M_avoid', [])}
        caution = {norm(x) for x in decision.get('M_caution', [])}
        alt = {norm(x) for x in decision.get('M_alt', [])}
        trace = state.get('trace_verification') or {}
        critic = state.get('critic_findings') or []
        if rec & avoid:
            tasks += ['safety_critique','decision_revision','trace_verification']
        if avoid and not alt:
            tasks += ['alternative_search','risk_benefit_deliberation','safety_critique','decision_revision','trace_verification']
        if trace.get('unsupported_claims'):
            tasks += ['safety_critique','decision_revision','trace_verification']
        if rec & caution:
            levels = decision.get('M_level', {}) or {}
            unresolved=True
            for med, level in levels.items():
                if norm(med) in caution and 'caution' in str(level).lower():
                    unresolved=False
            if unresolved:
                tasks += ['risk_benefit_deliberation','safety_critique','decision_revision','trace_verification']
        if any(f.get('severity') == 'critical' and not f.get('resolved') for f in critic):
            tasks += ['decision_revision','trace_verification']
        out=[]
        for t in tasks:
            if t not in out:
                out.append(t)
        return out
    def run_case(self, case):
        state = self._initialize_state(case)
        self._log(state, 'start', {'visible_agents': ['ClinicalStateGrounder','GuidelineCandidateAgent','DrugConflictAuditor','GeriatricRiskAuditor','AlternativeAgent','PMCGBuilder','RiskBenefitDeliberator','SafetyCritic','RevisionAgent','TraceVerifier'], 'agent_count': 10, 'max_rounds': self.max_rounds, 'packaging_note': 'Target-style visible wrappers preserve current outputs.'})
        for task in self._initial_tasks():
            state = self._run_task(state, task)
        for round_idx in range(1, self.max_rounds + 1):
            state['iteration'] = round_idx
            tasks = self._needs_review(state)
            if not tasks:
                self._log(state, 'converged', {'round': round_idx})
                break
            self._log(state, 'review_round', {'round': round_idx, 'tasks': tasks})
            for task in tasks:
                state = self._run_task(state, task)
        decision = state.get('decision') or {'M_rec': [], 'M_avoid': [], 'M_caution': [], 'M_alt': [], 'M_level': {}, 'U': True, 'uncertainty_reason': 'no decision generated'}
        pmcg = state.get('pmcg') or {'nodes': [], 'edges': [], 'schema': {'node_types': [], 'edge_types': []}}
        return {
            'case_id': case['case_id'],
            'final_decision': decision,
            'therapeutic_needs': state.get('therapeutic_needs', []),
            'open_candidate_generation': state.get('open_candidate_generation', {}),
            'safety_pruning': state.get('safety_pruning', {}),
            'bounded_secondary_outputs': state.get('bounded_secondary_outputs', {}),
            'pmcg': pmcg,
            'agent_messages': state['messages'],
            'agent_interactions': state['agent_interactions'],
            'blackboard_summary': {'keys': sorted(state.get('blackboard', {}).keys())},
            'yes_no_decision_board': state.get('yes_no_decision_board', {}),
            'yes_no_judgments': state.get('yes_no_judgments', []),
            'critic_findings': state.get('critic_findings', []),
            'orchestration_log': state['orchestration_log'],
            'reasoning_trace': [f"{m['agent']} -> {m['status']} -> {m['summary']}" for m in state['messages']],
            'evidence_sources': sorted({e.get('source_id') for e in pmcg.get('edges', []) if e.get('source_id')}),
            'risk_types': sorted({e.get('risk_type') for e in pmcg.get('edges', []) if e.get('effect') in ('avoid','caution') and e.get('risk_type')}),
            'trace_verification': state.get('trace_verification', {}),
        }
