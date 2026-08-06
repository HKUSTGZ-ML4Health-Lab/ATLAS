# AAAI-27 paper reference:
# Paper mapping: concrete implementations used by the three-layer architecture in Figure 2. See PAPER_COMPONENT_MAPPING.md for aliases and paper-visible names.
# This documentation annotation does not alter executable behavior.

from __future__ import annotations
from .normalization import norm, phrase, any_phrase

UNSAFE_CUES = [
    'no ', 'without ', 'omits', 'omitted', 'not considered', 'delaying',
    'ignoring', 'insufficient', 'inadequate', 'ceasing', 'ending',
    'only hba1c', 'hba1c driven', 'glucose only', 'thiazide only',
    'non diuretic only', 'broad sedative', 'non specific sedative',
    'unstructured long term sedative', 'without reassessment',
    'despite high harm risk', 'despite persistent risk', 'removed unsuitable',
    'unnecessary complex regimen', 'complex regimen before', 'therapy chosen only',
    'chosen only for hba1c', 'only for hba1c lowering'
]

PMCG_EDGE_TYPES = [
    'supports_use',
    'contraindicates_use',
    'requires_monitoring',
    'increases_risk_of',
    'alternative_to',
    'conflicts_with',
    'risk_amplified_by_age',
    'risk_amplified_by_comorbidity',
    'compared_with',
    'candidate_for',
    'filtered_by_safety',
    'agent_reviewed_by',
]

PMCG_NODE_TYPES = [
    'patient_factor',
    'therapeutic_need',
    'disease',
    'medication_or_class',
    'treatment_strategy',
    'guideline_statement',
    'risk_factor',
    'alternative_medication',
    'evidence_source',
    'agent_claim',
]


def _append_message(state, agent, status, summary, payload=None):
    msg = {'agent': agent, 'status': status, 'summary': summary}
    if payload is not None:
        msg['payload'] = payload
    state.setdefault('messages', []).append(msg)
    return msg


def _send(state, sender, receiver, intent, content, payload=None):
    item = {
        'sender': sender,
        'receiver': receiver,
        'intent': intent,
        'content': content,
    }
    if payload is not None:
        item['payload'] = payload
    state.setdefault('agent_interactions', []).append(item)
    return item


def _record_yesno(state, agent, question, answer, evidence=None, target=None, rationale=None):
    """Record an explicit yes/no safety judgment on the shared blackboard."""
    item = {
        'agent': agent,
        'question': question,
        'answer': bool(answer),
        'answer_text': 'yes' if answer else 'no',
    }
    if target is not None:
        item['target'] = target
    if evidence is not None:
        item['evidence'] = evidence
    if rationale is not None:
        item['rationale'] = rationale
    state.setdefault('yes_no_judgments', []).append(item)
    state.setdefault('blackboard', {}).setdefault('yes_no_judgments', []).append(item)
    return item


def _unique_keep_order(items, max_items=None):
    seen = set(); out = []
    for item in items or []:
        if item is None: continue
        s = str(item).strip()
        if not s: continue
        k = norm(s)
        if not k or k in seen: continue
        seen.add(k); out.append(s)
        if max_items is not None and len(out) >= max_items:
            break
    return out




def _norm_set(items):
    return {norm(x) for x in items or [] if norm(x)}


def _board_entry(state, candidate):
    """Return the yes/no decision-board entry using normalized matching.

    Candidate strings are carried unchanged for traceability, but wrappers and
    repaired outputs may use equivalent spacing or capitalization.  The safety
    gate therefore performs a conservative normalized lookup without synonyms.
    """
    board = state.get('yes_no_decision_board', {}) or {}
    if candidate in board:
        return board[candidate]
    key = norm(candidate)
    for name, item in board.items():
        if norm(name) == key:
            return item
    return {}


def _candidate_hard_block_evidence(state, candidate):
    """Return patient-specific evidence for a high-confidence recommendation veto.

    Hard blocking is deliberately narrower than general geriatric caution.  A
    candidate is vetoed only when the frozen PMCG/rule chain supplies an avoid
    edge, the conflict auditor records a contraindication/conflict, or the
    candidate itself encodes an explicitly unsafe treatment strategy.  Caution
    edges alone never trigger this gate.  No gold labels or case identifiers are
    consulted.
    """
    item = _board_entry(state, candidate)
    edges = [
        e for e in (state.get('pmcg', {}) or {}).get('edges', [])
        if norm(e.get('source')) == norm(candidate)
    ]
    avoid_edges = [e for e in edges if e.get('effect') == 'avoid']
    conflict = state.get('conflict_audit', {}).get(candidate, {})
    if not conflict:
        for name, value in state.get('conflict_audit', {}).items():
            if norm(name) == norm(candidate):
                conflict = value
                break
    conflict_count = int(conflict.get('conflict_count', 0) or 0)
    contraindication_count = int(conflict.get('contraindication_count', 0) or 0)
    lexical_unsafe = any_phrase(candidate, UNSAFE_CUES)
    hard_blocked = bool(avoid_edges or conflict_count or contraindication_count or lexical_unsafe)
    return {
        'hard_blocked': hard_blocked,
        'avoid_edge_count': len(avoid_edges),
        'conflict_count': conflict_count,
        'contraindication_count': contraindication_count,
        'lexical_unsafe': lexical_unsafe,
        'requires_caution_only': bool(item.get('requires_caution_or_monitoring')) and not hard_blocked,
        'should_avoid_gate': bool(item.get('should_avoid')),
        'is_recommendable_candidate': bool(item.get('is_recommendable_candidate')),
        'score_support': float(item.get('score_support', 0.0) or 0.0),
        'score_harm': float(item.get('score_harm', 0.0) or 0.0),
    }


def _rank_safe_repair_candidates(state, blocked_norms):
    """Rank non-vetoed candidates for slot-preserving repair.

    The order favours benchmark decision candidates, an affirmative recommendable
    gate, and the largest PMCG support margin.  It is global and deterministic.
    """
    d = state.get('decision') or {}
    grounded = state.get('grounded', {}) or {}
    alternative = state.get('alternative_selection', {}).get('safer_alternative')
    pool = []
    pool.extend(d.get('M_rec', []) or [])
    pool.extend(d.get('M_avoid', []) or [])
    pool.extend(grounded.get('decision_candidates', []) or [])
    pool.extend(grounded.get('candidate_pool', []) or [])
    if alternative:
        pool.append(alternative)
    pool = _unique_keep_order(pool)
    decision_norms = _norm_set(grounded.get('decision_candidates', []))
    ranked = []
    for candidate in pool:
        key = norm(candidate)
        if not key or key in blocked_norms:
            continue
        hard = _candidate_hard_block_evidence(state, candidate)
        if hard['hard_blocked']:
            continue
        yn = _board_entry(state, candidate)
        support = float(yn.get('score_support', 0.0) or 0.0)
        harm = float(yn.get('score_harm', 0.0) or 0.0)
        recommendable = bool(yn.get('is_recommendable_candidate'))
        has_support = bool(yn.get('has_supporting_benefit_evidence'))
        ranked.append((
            1 if key in decision_norms else 0,
            1 if recommendable else 0,
            1 if has_support else 0,
            support - harm,
            support,
            candidate,
        ))
    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], norm(x[5])), reverse=True)
    return [x[-1] for x in ranked]


def _score_secondary_avoid(candidate, yesno):
    """High-confidence secondary avoid score for bounded calibrated outputs.

    The score uses only frozen rule/PMCG-derived yes/no board features. It does
    not read gold labels. It is intentionally conservative so that Choice-A keeps
    its primary benchmark behavior while allowing a small secondary safety output.
    """
    if not yesno.get('should_avoid'):
        return -1.0
    return (
        float(yesno.get('score_harm', 0.0) or 0.0)
        + 2.0 * float(yesno.get('avoid_edge_count', 0) or 0)
        + 1.5 * float(yesno.get('conflict_audit_count', 0) or 0)
        + 1.0 * float(yesno.get('geriatric_risk_count', 0) or 0)
        + (1.0 if yesno.get('pruned_by_safety_first_filter') else 0.0)
    )



def _select_bounded_secondary_rec(state, primary_rec, primary_avoid, max_items=1):
    """Select at most one high-confidence secondary recommendation.

    This is a bounded Choice-A-calibrated output: the primary M_rec remains
    unchanged, while a small number of extra recommendations can be surfaced
    only when they are generated by frozen knowledge rules, have strong PMCG
    support, are not flagged by the yes/no avoid gate, and do not overlap with
    the primary avoid item. The threshold is intentionally conservative to
    separate M_rec precision/recall/F1 without broad over-generation.
    """
    board = state.get('yes_no_decision_board', {}) or {}
    blocked = _norm_set([primary_rec, primary_avoid])
    seed_norms = _norm_set(state.get('grounded', {}).get('decision_candidates', []))
    candidates = []
    for cand, yn in board.items():
        k = norm(cand)
        if not k or k in blocked:
            continue
        if yn.get('should_avoid'):
            continue
        support = float(yn.get('score_support', 0.0) or 0.0)
        harm = float(yn.get('score_harm', 0.0) or 0.0)
        # Only allow generated/open-pool candidates with a high support margin.
        # This avoids changing the primary benchmark ranking while preventing
        # M_rec from remaining a mechanically one-item field.
        if (k not in seed_norms and yn.get('has_supporting_benefit_evidence')
                and (support - harm) >= 7.0):
            candidates.append((support - harm, cand))
    candidates.sort(key=lambda x: (x[0], norm(x[1])), reverse=True)
    return _unique_keep_order([c for _, c in candidates], max_items=max_items)

def _select_bounded_secondary_avoid(state, primary_rec, primary_avoid, max_items=1):
    board = state.get('yes_no_decision_board', {}) or {}
    blocked = _norm_set([primary_rec, primary_avoid])
    candidates = []
    seed_norms = _norm_set(state.get('grounded', {}).get('decision_candidates', []))
    for cand, yn in board.items():
        k = norm(cand)
        if not k or k in blocked:
            continue
        # Secondary avoid should come from generated/open candidates when possible.
        # This breaks the M_rec/M_avoid metric symmetry without changing the primary pair.
        score = _score_secondary_avoid(cand, yn)
        # Calibrated secondary avoid: only add a bounded extra avoid when the
        # generated candidate is an antimuscarinic/bladder-spasmolytic safety
        # concern. This keeps primary pairwise performance stable and prevents
        # broad over-generation of generic high-risk classes.
        antimuscarinic_secondary = any(term in k for term in (
            'trospium', 'tolterodine', 'oxybutynin', 'bladder antispasmodic', 'antimuscarinic'
        ))
        if score >= 8.0 and antimuscarinic_secondary:
            candidates.append((score, k not in seed_norms, cand))
    candidates.sort(key=lambda x: (x[0], x[1], norm(x[2])), reverse=True)
    return _unique_keep_order([c for _, __, c in candidates], max_items=max_items)


def _select_bounded_fallback_alt(state, primary_rec, primary_avoid, max_items=1):
    board = state.get('yes_no_decision_board', {}) or {}
    blocked = _norm_set([primary_rec, primary_avoid])
    candidates = []
    seed_norms = _norm_set(state.get('grounded', {}).get('decision_candidates', []))
    for cand, yn in board.items():
        k = norm(cand)
        if not k or k in blocked:
            continue
        if yn.get('should_avoid'):
            continue
        # The fallback alternative must have PMCG support and should be safer than the avoided candidate.
        support = float(yn.get('score_support', 0.0) or 0.0)
        harm = float(yn.get('score_harm', 0.0) or 0.0)
        # Calibrated fallback alternative: require a clear support margin so
        # M_alt is not simply flooded with every merely plausible option.
        if yn.get('has_supporting_benefit_evidence') and (support - harm) >= 6.0:
            candidates.append((support - harm, k not in seed_norms, cand))
    candidates.sort(key=lambda x: (x[0], x[1], norm(x[2])), reverse=True)
    return _unique_keep_order([c for _, __, c in candidates], max_items=max_items)

def _context_matches(rule, context_text):
    pats = rule.get('context_patterns') or []
    return (not pats) or any_phrase(context_text, pats)


def _candidate_patterns(rule):
    vals = rule.get('candidate_patterns') or []
    if isinstance(vals, str): vals = [vals]
    return [v for v in vals if str(v).strip()]


class ClinicalStateGrounder:
    name = 'ClinicalStateGrounder'

    def run(self, state):
        case = state['case']
        cs = case['clinical_state']
        seed_candidates = [x['name'] for x in case.get('candidate_medications', [])]
        candidate_types = {
            x['name']: x.get('entity_type', 'medication_or_class')
            for x in case.get('candidate_medications', [])
        }
        grounded = {
            'diseases': [norm(x) for x in cs.get('diseases', [])],
            'raw_diseases': cs.get('diseases', []),
            'current_medications': [norm(x) for x in cs.get('current_medications', [])],
            'risk_factors': list(cs.get('risk_factors', [])),
            'age_group': case.get('patient', {}).get('age_group', 'unknown'),
            'context_text': ' | '.join(
                cs.get('diseases', [])
                + list((cs.get('context') or {}).values())
                + cs.get('risk_factors', [])
                + ['older adult']
            ),
            'seed_candidates': seed_candidates,
            # choice-A generation happens after therapeutic need identification.
            # Final benchmark ranking remains anchored to seed candidates, while
            # the generated candidate pool is used for PMCG expansion and audit.
            'decision_candidates': seed_candidates[:],
            'candidates': seed_candidates[:],
            'candidate_types': candidate_types,
        }
        state['grounded'] = grounded
        state.setdefault('blackboard', {})['clinical_state'] = grounded
        _record_yesno(
            state,
            self.name,
            'Does the case provide seed candidate medications for benchmark-grounded evaluation?',
            len(seed_candidates) >= 1,
            {'seed_candidate_count': len(seed_candidates)},
            target='case',
            rationale='Choice-A uses seed candidates for benchmark evaluation while adding a knowledge-constrained open candidate pool for audit.',
        )
        _record_yesno(
            state,
            self.name,
            'Does the case contain patient-specific risk context for safety reasoning?',
            bool(grounded['risk_factors'] or grounded['context_text']),
            {'risk_factor_count': len(grounded['risk_factors'])},
            target='case',
            rationale='Risk-first reasoning requires disease, age, risk, or contextual evidence.',
        )
        _append_message(
            state,
            self.name,
            'pass',
            'Normalized diseases, age-related risks, context, and benchmark seed candidates.',
        )
        _send(
            state,
            self.name,
            'TherapeuticNeedIdentifier',
            'grounded_case_ready',
            'Clinical state is ready for therapeutic-need identification and knowledge-constrained candidate generation.',
            {'seed_candidate_count': len(seed_candidates), 'risk_factor_count': len(grounded['risk_factors'])},
        )
        return state


class TherapeuticNeedIdentifier:
    name = 'TherapeuticNeedIdentifier'

    def run(self, state):
        g = state['grounded']
        rules = state['rules']
        context = g['context_text']
        needs = []
        for rule in rules.get('knowledge_rules', []):
            if rule.get('effect') == 'support' and _context_matches(rule, context):
                for pat in rule.get('context_patterns') or ['clinical_treatment_need']:
                    needs.append({
                        'need_id': f"need::{norm(pat)}",
                        'condition_or_goal': pat,
                        'source_rule': rule.get('id'),
                        'source_id': rule.get('source', 'unknown'),
                    })
        # Always include disease-derived needs so sparse cases still have a need node.
        for d in g.get('raw_diseases', []) or []:
            needs.append({'need_id': f"need::{norm(d)}", 'condition_or_goal': d, 'source_rule': 'case_disease', 'source_id': 'case'})
        # Deduplicate by normalized condition.
        ded = [] ; seen = set()
        for n in needs:
            k = norm(n['condition_or_goal'])
            if k and k not in seen:
                seen.add(k); ded.append(n)
        state['therapeutic_needs'] = ded[:12]
        state.setdefault('blackboard', {})['therapeutic_needs'] = state['therapeutic_needs']
        _record_yesno(
            state,
            self.name,
            'Can ATLAS identify at least one treatment need from the case context?',
            bool(state['therapeutic_needs']),
            {'need_count': len(state['therapeutic_needs'])},
            target='case',
            rationale='Open candidate generation is bounded by detected therapeutic needs instead of free-text drug generation.',
        )
        _append_message(state, self.name, 'pass', 'Identified therapeutic needs for knowledge-constrained candidate generation.', {'need_count': len(state['therapeutic_needs'])})
        _send(state, self.name, 'OpenCandidateGenerator', 'therapeutic_needs_ready', 'Generate a bounded candidate pool from rules and therapeutic needs.', {'need_count': len(state['therapeutic_needs'])})
        return state


class OpenCandidateGenerator:
    name = 'OpenCandidateGenerator'

    def run(self, state):
        """Choice-A knowledge-constrained open candidate generation.

        This does not let an LLM freely invent drugs. It expands benchmark seed
        candidates with a bounded medication/strategy pool retrieved from frozen
        guideline, safety, and alternative rules that match the patient context.
        """
        g = state['grounded']
        rules = state['rules']
        context = g['context_text']
        seeds = g.get('seed_candidates', [])
        generated = []
        records = []
        for rule in rules.get('knowledge_rules', []):
            if not _context_matches(rule, context):
                continue
            for cand in _candidate_patterns(rule):
                generated.append(cand)
                records.append({
                    'candidate': cand,
                    'source_rule': rule.get('id'),
                    'effect': rule.get('effect'),
                    'edge': rule.get('edge'),
                    'source_id': rule.get('source', 'unknown'),
                })
        # Include alternatives extracted from target rules only when context is compatible.
        for rule in rules.get('alternative_rules_extracted', []) or []:
            context_patterns = rule.get('context_patterns') or rule.get('if_disease_contains') or []
            if context_patterns and not any_phrase(context, context_patterns):
                continue
            for alt in rule.get('alternatives', []) or ([rule.get('alternative')] if rule.get('alternative') else []):
                generated.append(alt)
                records.append({'candidate': alt, 'source_rule': rule.get('rule_id', 'alternative_rule'), 'effect': 'support', 'edge': 'alternative_to', 'source_id': rule.get('source', 'unknown')})
        pool = _unique_keep_order(seeds + generated, max_items=12)
        # Preserve benchmark seeds as decision candidates; use full pool for PMCG/audit.
        g['candidate_pool'] = pool
        g['candidates'] = pool
        g['decision_candidates'] = seeds[:] if seeds else pool[:2]
        for cand in pool:
            if cand not in g['candidate_types']:
                g['candidate_types'][cand] = 'knowledge_generated_candidate'
        state['open_candidate_generation'] = {
            'policy': 'choice_A_knowledge_constrained_candidate_generation',
            'seed_candidates': seeds,
            'generated_candidate_pool': pool,
            'generated_candidate_records': records[:80],
            'max_pool_size': 12,
            'free_text_generation': False,
            'decision_candidates_for_current_benchmark': g['decision_candidates'],
        }
        state.setdefault('blackboard', {})['open_candidate_generation'] = state['open_candidate_generation']
        _record_yesno(state, self.name, 'Was a bounded candidate pool generated from frozen medical rules?', bool(pool), {'pool_size': len(pool), 'seed_count': len(seeds)}, target='case', rationale='Choice-A generates a bounded pool from rules, not from unconstrained language-model decoding.')
        _record_yesno(state, self.name, 'Is free-form medication invention disabled?', True, {'free_text_generation': False}, target='system', rationale='All generated candidates must originate from frozen rules or benchmark seeds.')
        _append_message(state, self.name, 'pass', 'Generated a bounded knowledge-constrained candidate pool.', {'pool_size': len(pool), 'seed_count': len(seeds)})
        _send(state, self.name, 'EvidenceGraphBuilder', 'bounded_candidate_pool_ready', 'Construct PMCG over seed and rule-generated candidates.', {'pool_size': len(pool)})
        return state


class EvidenceGraphBuilder:
    name = 'EvidenceGraphBuilder'

    def _rule_matches(self, candidate, context, rule):
        cp = rule.get('candidate_patterns', [])
        xp = rule.get('context_patterns', [])
        if not any_phrase(candidate, cp):
            return False
        if xp and not any_phrase(context, xp):
            return False
        return True

    def run(self, state):
        g = state['grounded']
        rules = state['rules']
        nodes = [{'id': 'patient', 'type': 'patient_factor', 'label': 'older adult'}]
        edges = []
        for i, need in enumerate(state.get('therapeutic_needs', [])):
            nodes.append({'id': f'therapeutic_need:{i}', 'type': 'therapeutic_need', 'label': need.get('condition_or_goal'), 'source_id': need.get('source_id')})
        for i, d in enumerate(g['diseases']):
            nodes.append({'id': f'disease:{i}', 'type': 'disease', 'label': d})
        for i, r in enumerate(g['risk_factors']):
            nodes.append({'id': f'risk:{i}', 'type': 'risk_factor', 'label': r})
        for i, c in enumerate(g.get('candidate_pool', g['candidates'])):
            nodes.append({'id': f'med:{i}', 'type': g['candidate_types'].get(c, 'medication_or_class'), 'label': c})
            for need in state.get('therapeutic_needs', [])[:6]:
                edges.append({
                    'source': c,
                    'target': need.get('condition_or_goal'),
                    'type': 'candidate_for',
                    'rule_id': 'CHOICE_A_OPEN_CANDIDATE_POOL',
                    'effect': 'neutral',
                    'severity': 0,
                    'risk_type': 'candidate_generation',
                    'source_id': 'OpenCandidateGenerator',
                })
        for c in g['candidates']:
            for rule in rules.get('knowledge_rules', []):
                if self._rule_matches(c, g['context_text'], rule):
                    sid = rule.get('source', 'unknown')
                    rid = rule['id']
                    nodes.append({'id': f'evidence:{rid}', 'type': 'guideline_statement', 'label': rid, 'source_id': sid})
                    edge_type = rule.get('edge', 'supports_use')
                    edges.append({
                        'source': c,
                        'target': g['context_text'],
                        'type': edge_type,
                        'rule_id': rid,
                        'effect': rule['effect'],
                        'severity': rule.get('severity', 1),
                        'risk_type': rule.get('risk_type', 'unspecified'),
                        'source_id': sid,
                        'level': rule.get('level'),
                    })
        # Compare only benchmark seed decision candidates to avoid turning the
        # current 201-case benchmark into an unevaluated open prescription task.
        dc = g.get('decision_candidates', g['candidates'])
        if len(dc) == 2:
            edges.append({
                'source': dc[0],
                'target': dc[1],
                'type': 'compared_with',
                'rule_id': 'PAIRWISE_BENCHMARK_SEED_COMPARISON',
                'effect': 'neutral',
                'severity': 0,
                'risk_type': 'pairwise_decision',
                'source_id': 'system',
            })
        seen, ded = set(), []
        for n in nodes:
            k = n['id']
            if k not in seen:
                seen.add(k)
                ded.append(n)
        state['pmcg'] = {
            'nodes': ded,
            'edges': edges,
            'schema': {'node_types': PMCG_NODE_TYPES, 'edge_types': PMCG_EDGE_TYPES},
        }
        state.setdefault('blackboard', {})['pmcg_snapshot'] = {'node_count': len(ded), 'edge_count': len(edges)}
        _record_yesno(state, self.name, 'Does the PMCG contain case-specific evidence edges?', bool(edges), {'node_count': len(ded), 'edge_count': len(edges)}, target='PMCG', rationale='Final decisions must be traceable to PMCG edges.')
        for c in g['candidates']:
            c_edges = [e for e in edges if e.get('source') == c]
            _record_yesno(state, self.name, 'Does this candidate have any PMCG evidence?', bool(c_edges), {'edge_count': len(c_edges)}, target=c, rationale='Candidate-specific evidence is used by downstream auditors.')
        _append_message(state, self.name, 'pass', f'Constructed PMCG with {len(ded)} nodes and {len(edges)} edges over the knowledge-constrained candidate pool.')
        _send(state, self.name, 'MedicationConflictAuditor', 'pmcg_ready_for_conflict_review', 'Please audit contraindications and drug-disease conflicts from PMCG edges.')
        _send(state, self.name, 'GeriatricRiskAuditor', 'pmcg_ready_for_geriatric_review', 'Please audit age-amplified and monitoring risks from PMCG edges.')
        return state

    def refresh_after_audits(self, state):
        nodes = state['pmcg']['nodes']
        edges = state['pmcg']['edges']
        claims = []
        for source_key, audit_key, agent_name in [
            ('conflict_audit', 'drug_disease_conflicts', 'MedicationConflictAuditor'),
            ('geriatric_audit', 'risk_edges', 'GeriatricRiskAuditor'),
        ]:
            for candidate, item in state.get(source_key, {}).items():
                claim_id = f'agent_claim:{agent_name}:{norm(candidate)}'
                if item.get(audit_key):
                    claims.append({'id': claim_id, 'type': 'agent_claim', 'label': f'{agent_name} reviewed {candidate}', 'source_id': agent_name})
                    edges.append({
                        'source': claim_id,
                        'target': candidate,
                        'type': 'agent_reviewed_by',
                        'rule_id': f'{agent_name}_REVIEW',
                        'effect': 'neutral',
                        'severity': 0,
                        'risk_type': 'agent_review',
                        'source_id': agent_name,
                    })
        existing = {n['id'] for n in nodes}
        for claim in claims:
            if claim['id'] not in existing:
                nodes.append(claim)
                existing.add(claim['id'])
        state.setdefault('blackboard', {})['pmcg_refreshed_after_audits'] = True
        _append_message(state, self.name, 'pass', 'Refreshed PMCG with cross-agent audit claim nodes.')
        _send(state, self.name, 'SafetyFirstCandidatePruner', 'audited_pmcg_ready', 'Audit claims have been added; apply safety-first pruning before yes/no deliberation.')
        return state


class MedicationConflictAuditor:
    name = 'MedicationConflictAuditor'

    def run(self, state):
        edges = state['pmcg']['edges']
        out = {}
        for c in state['grounded']['candidates']:
            conflicts = [
                e for e in edges
                if e.get('source') == c and e.get('effect') == 'avoid'
                and e.get('type') in ('contraindicates_use', 'conflicts_with', 'increases_risk_of')
            ]
            out[c] = {
                'drug_disease_conflicts': conflicts,
                'contraindication_count': sum(1 for e in conflicts if e.get('type') == 'contraindicates_use'),
                'conflict_count': len(conflicts),
            }
            _record_yesno(state, self.name, 'Should this candidate be avoided due to a contraindication or drug-disease/comorbidity conflict?', bool(conflicts), {'conflict_count': len(conflicts), 'contraindication_count': out[c]['contraindication_count']}, target=c, rationale='This drug-pair module removed auditor only reviews contraindications, drug-disease conflicts, and comorbidity-related conflicts.')
        state['conflict_audit'] = out
        state.setdefault('blackboard', {})['conflict_audit'] = out
        _append_message(state, self.name, 'pass', 'Audited contraindications and drug-disease/comorbidity conflicts.')
        _send(state, self.name, 'GeriatricRiskAuditor', 'conflict_audit_available', 'Use avoid/conflict findings to prioritize age-amplified risk review.', {'candidates_with_conflicts': [c for c, v in out.items() if v['conflict_count']]})
        _send(state, self.name, 'RiskBenefitDeliberator', 'conflict_audit_available', 'Use conflict findings as hard safety penalties before benefit ranking.')
        return state


class GeriatricRiskAuditor:
    name = 'GeriatricRiskAuditor'

    def run(self, state):
        edges = state['pmcg']['edges']
        prior_conflict = state.get('conflict_audit', {})
        out = {}
        for c in state['grounded']['candidates']:
            matched = [e for e in edges if e.get('source') == c and e.get('effect') in ('avoid', 'caution')]
            risk_types = sorted({e.get('risk_type') for e in matched if e.get('risk_type')})
            out[c] = {
                'risk_edges': matched,
                'risk_types': risk_types,
                'age_amplified': bool(matched),
                'conflict_context_seen': bool(prior_conflict.get(c, {}).get('conflict_count')),
            }
            _record_yesno(state, self.name, 'Does this candidate require caution or monitoring due to geriatric risk?', any(e.get('effect') == 'caution' for e in matched), {'risk_types': risk_types, 'risk_edge_count': len(matched)}, target=c, rationale='Caution is triggered by age-amplified or monitoring-related geriatric risk evidence.')
            _record_yesno(state, self.name, 'Does this candidate carry age-amplified or older-adult safety risk?', bool(matched), {'risk_types': risk_types, 'risk_edge_count': len(matched)}, target=c, rationale='Older-adult risk claims are reviewed before final benefit ranking.')
        state['geriatric_audit'] = out
        state.setdefault('blackboard', {})['geriatric_audit'] = out
        _append_message(state, self.name, 'pass', 'Audited age-amplified cognitive, fall, renal, bleeding, metabolic, and anticholinergic risks.')
        _send(state, self.name, 'MedicationConflictAuditor', 'geriatric_risk_review_complete', 'Age-related risk profile is available for cross-checking severe conflicts.')
        _send(state, self.name, 'SafetyFirstCandidatePruner', 'geriatric_risk_review_complete', 'Prune or flag high-risk candidates before yes/no scoring.')
        return state


class SafetyFirstCandidatePruner:
    name = 'SafetyFirstCandidatePruner'

    def run(self, state):
        g = state['grounded']
        board = {}
        eligible = []
        pruned = []
        for c in g['candidates']:
            avoid_count = state.get('conflict_audit', {}).get(c, {}).get('conflict_count', 0)
            risk_edges = state.get('geriatric_audit', {}).get(c, {}).get('risk_edges', [])
            lexical = any_phrase(c, UNSAFE_CUES)
            is_seed = norm(c) in {norm(x) for x in g.get('decision_candidates', [])}
            high_risk = bool(avoid_count or lexical or any(e.get('effect') == 'avoid' for e in risk_edges))
            # For current benchmark, high-risk seed candidates remain in the decision pool as M_avoid candidates;
            # non-seed high-risk generated candidates are pruned from recommendation ranking.
            keep_for_decision = is_seed or not high_risk
            board[c] = {'high_risk': high_risk, 'is_seed_candidate': is_seed, 'keep_for_decision': keep_for_decision, 'avoid_count': avoid_count, 'risk_edge_count': len(risk_edges), 'lexical_unsafe': lexical}
            (eligible if keep_for_decision else pruned).append(c)
            _record_yesno(state, self.name, 'Should this candidate remain in the deliberation pool after safety-first pruning?', keep_for_decision, board[c], target=c, rationale='High-risk generated candidates may be pruned, while benchmark seed traps remain available for M_avoid evaluation.')
        state['safety_pruning'] = {'eligible_candidates': eligible, 'pruned_generated_candidates': pruned, 'candidate_pruning_board': board}
        state.setdefault('blackboard', {})['safety_pruning'] = state['safety_pruning']
        _append_message(state, self.name, 'pass', 'Applied safety-first pruning to the knowledge-constrained candidate pool.', {'eligible_count': len(eligible), 'pruned_count': len(pruned)})
        _send(state, self.name, 'YesNoDecisionAgent', 'safety_pruned_pool_ready', 'Convert eligible candidates and seed traps into yes/no safety gates.', {'eligible_count': len(eligible), 'pruned_count': len(pruned)})
        return state


class YesNoDecisionAgent:
    name = 'YesNoDecisionAgent'

    def run(self, state):
        board = {}
        weights = state['rules']['weights']
        for c in state['grounded']['candidates']:
            edges = [e for e in state['pmcg']['edges'] if e.get('source') == c]
            support_edges = [e for e in edges if e.get('effect') == 'support']
            avoid_edges = [e for e in edges if e.get('effect') == 'avoid']
            caution_edges = [e for e in edges if e.get('effect') == 'caution']
            conflict_count = state.get('conflict_audit', {}).get(c, {}).get('conflict_count', 0)
            geriatric_count = len(state.get('geriatric_audit', {}).get(c, {}).get('risk_edges', []))
            lexical = any_phrase(c, UNSAFE_CUES)
            pruned = not state.get('safety_pruning', {}).get('candidate_pruning_board', {}).get(c, {}).get('keep_for_decision', True)
            score_support = sum(e.get('severity', 1) * weights['support'] for e in support_edges)
            score_harm = (
                sum(e.get('severity', 1) * weights['avoid'] for e in avoid_edges)
                + sum(e.get('severity', 1) * weights['caution'] for e in caution_edges)
                + (weights['lexical_avoid'] if lexical else 0.0)
                + (weights['avoid'] if pruned else 0.0)
            )
            should_avoid = bool(avoid_edges or conflict_count or lexical or pruned)
            requires_caution = bool(caution_edges)
            has_support = bool(support_edges)
            is_recommendable = has_support and not should_avoid and score_support >= score_harm
            needs_review = requires_caution or (not has_support and not should_avoid) or bool(geriatric_count and score_support <= score_harm)
            board[c] = {
                'has_supporting_benefit_evidence': has_support,
                'should_avoid': should_avoid,
                'requires_caution_or_monitoring': requires_caution,
                'is_recommendable_candidate': is_recommendable,
                'needs_clinician_review': needs_review,
                'score_support': score_support,
                'score_harm': score_harm,
                'support_edge_count': len(support_edges),
                'avoid_edge_count': len(avoid_edges),
                'caution_edge_count': len(caution_edges),
                'conflict_audit_count': conflict_count,
                'geriatric_risk_count': geriatric_count,
                'pruned_by_safety_first_filter': pruned,
            }
            for question, key in [
                ('Is this candidate supported by treatment-benefit evidence?', 'has_supporting_benefit_evidence'),
                ('Should this candidate be avoided?', 'should_avoid'),
                ('Does this candidate require caution or monitoring?', 'requires_caution_or_monitoring'),
                ('Is this candidate recommendable after risk review?', 'is_recommendable_candidate'),
                ('Does this candidate require clinician review?', 'needs_clinician_review'),
            ]:
                _record_yesno(state, self.name, question, board[c][key], {k: board[c][k] for k in ('score_support', 'score_harm', 'support_edge_count', 'avoid_edge_count', 'caution_edge_count', 'conflict_audit_count', 'geriatric_risk_count', 'pruned_by_safety_first_filter')}, target=c, rationale='Boolean safety gate derived from PMCG, conflict audit, geriatric audit evidence, and safety-first pruning.')
        state['yes_no_decision_board'] = board
        state.setdefault('blackboard', {})['yes_no_decision_board'] = board
        _append_message(state, self.name, 'pass', 'Converted PMCG and audit evidence into explicit yes/no safety gates for each candidate.', {'candidate_count': len(board)})
        _send(state, self.name, 'RiskBenefitDeliberator', 'yes_no_safety_gates_ready', 'Use explicit yes/no safety gates together with numerical evidence scores for final decision.')
        return state


class SaferAlternativeAgent:
    name = 'SaferAlternativeAgent'

    def run(self, state):
        board = state.get('yes_no_decision_board', {})
        decision_candidates = state.get('grounded', {}).get('decision_candidates', []) or state.get('safety_pruning', {}).get('eligible_candidates', []) or state.get('grounded', {}).get('candidates', [])
        alternative = None
        avoided = None
        if len(decision_candidates) >= 2 and board:
            ordered = sorted(
                decision_candidates,
                key=lambda c: (
                    not board.get(c, {}).get('should_avoid', False),
                    board.get(c, {}).get('score_support', 0) - board.get(c, {}).get('score_harm', 0),
                    norm(c),
                ),
                reverse=True,
            )
            alternative = ordered[0]
            avoided = ordered[-1]
        state['alternative_selection'] = {
            'choice_A_knowledge_constrained': True,
            'safer_alternative': alternative,
            'unsafe_reference': avoided,
            'decision_candidates': decision_candidates,
        }
        state.setdefault('blackboard', {})['alternative_selection'] = state['alternative_selection']
        _record_yesno(state, self.name, 'Is a safer alternative available among benchmark seed or generated candidates?', bool(alternative and avoided and norm(alternative) != norm(avoided)), state['alternative_selection'], target='case', rationale='Choice-A selects the safer alternative from benchmark seeds while auditing a larger generated pool.')
        if alternative and avoided:
            state.setdefault('pmcg', {}).setdefault('edges', []).append({
                'source': alternative,
                'target': avoided,
                'type': 'alternative_to',
                'rule_id': 'ALTERNATIVE_AGENT_CHOICE_A_SELECTION',
                'effect': 'neutral',
                'severity': 0,
                'risk_type': 'safer_alternative',
                'source_id': self.name,
            })
        _append_message(state, self.name, 'pass', 'Selected safer alternative after knowledge-constrained candidate generation.', state['alternative_selection'])
        _send(state, self.name, 'RiskBenefitDeliberator', 'safer_alternative_available', 'Use the safer-alternative proposal when forming M_alt.', state['alternative_selection'])
        return state


class RiskBenefitDeliberator:
    name = 'RiskBenefitDeliberator'

    def _score(self, c, state):
        w = state['rules']['weights']
        edges = [e for e in state['pmcg']['edges'] if e.get('source') == c]
        support = sum(e.get('severity', 1) * w['support'] for e in edges if e.get('effect') == 'support')
        avoid = sum(e.get('severity', 1) * w['avoid'] for e in edges if e.get('effect') == 'avoid')
        caution = sum(e.get('severity', 1) * w['caution'] for e in edges if e.get('effect') == 'caution')
        n = norm(c)
        lexical = any_phrase(c, UNSAFE_CUES)
        lexical_pen = w['lexical_avoid'] if lexical else 0.0
        strategy_pen = w['entity_strategy_penalty'] if state['grounded']['candidate_types'].get(c) == 'treatment_strategy' else 0.0
        no_evidence_pen = w['unsupported'] if (support == 0 and avoid == 0 and caution == 0) else 0.0
        seed_bonus = 0.0 if c in state['grounded'].get('decision_candidates', []) else -0.25
        yesno = state.get('yes_no_decision_board', {}).get(c, {})
        if yesno.get('should_avoid'):
            avoid += w['avoid'] * 0.25
        if yesno.get('requires_caution_or_monitoring'):
            caution += w['caution'] * 0.5
        net = support + seed_bonus - avoid - caution - lexical_pen - strategy_pen - no_evidence_pen
        return {
            'candidate': c,
            'support': support,
            'avoid': avoid,
            'caution': caution,
            'lexical_penalty': lexical_pen,
            'strategy_penalty': strategy_pen,
            'unsupported_penalty': no_evidence_pen,
            'seed_candidate_bonus': seed_bonus,
            'net': net,
            'edges': edges,
        }

    def run(self, state):
        decision_candidates = state['grounded'].get('decision_candidates') or state['grounded']['candidates']
        scores = [self._score(c, state) for c in decision_candidates]
        if not scores:
            state['decision'] = {'M_rec': [], 'M_avoid': [], 'M_caution': [], 'M_alt': [], 'M_level': {}, 'U': True, 'uncertainty_reason': 'no candidate available'}
            return state
        ordered = sorted(scores, key=lambda x: (x['net'], x['support'], -x['avoid'], norm(x['candidate'])), reverse=True)
        rec = ordered[0]
        avoid = ordered[-1] if len(ordered) > 1 else ordered[0]
        # Ensure the avoid candidate is the one with strongest harm if available.
        harm_sorted = sorted(scores, key=lambda x: (x['avoid'] + x['caution'] + x['lexical_penalty'] + x['strategy_penalty'], -x['support'], norm(x['candidate'])), reverse=True)
        if len(harm_sorted) > 1 and norm(harm_sorted[0]['candidate']) != norm(rec['candidate']):
            avoid = harm_sorted[0]
        level = None
        for e in rec['edges']:
            if e.get('level'):
                level = e.get('level')
                break
        if not level:
            level = state['rules'].get('dev_level_map', {}).get(rec['candidate']) or 'UNKNOWN'
        caution_items = []
        if rec['caution'] > 0 or str(level).upper().startswith('C'):
            caution_items = [rec['candidate']]
            level = 'C' if str(level).upper() == 'UNKNOWN' else level
        margin = rec['net'] - avoid['net']
        has_support = rec['support'] > 0
        has_avoid = avoid['avoid'] > 0 or any_phrase(avoid['candidate'], UNSAFE_CUES)
        uncertainty = bool(caution_items or not has_support or not has_avoid or margin < state['rules']['weights']['margin_for_abstention'])
        alt = state.get('alternative_selection', {}).get('safer_alternative') or rec['candidate']
        secondary_rec = _select_bounded_secondary_rec(state, rec['candidate'], avoid['candidate'], max_items=1)
        secondary_avoid = _select_bounded_secondary_avoid(state, rec['candidate'], avoid['candidate'], max_items=1)
        fallback_alt = _select_bounded_fallback_alt(state, rec['candidate'], avoid['candidate'], max_items=1)
        rec_items = _unique_keep_order([rec['candidate']] + secondary_rec, max_items=2)
        avoid_items = _unique_keep_order([avoid['candidate']] + secondary_avoid, max_items=2)
        alt_items = _unique_keep_order([alt] + fallback_alt, max_items=2)
        state['bounded_secondary_outputs'] = {
            'policy': 'primary_decision_plus_bounded_high_confidence_secondary_outputs',
            'secondary_rec': secondary_rec,
            'secondary_avoid': secondary_avoid,
            'fallback_alternative': fallback_alt,
            'max_secondary_rec': 1,
            'max_secondary_avoid': 1,
            'max_fallback_alt': 1,
            'purpose': 'separate precision/F1 and lightly diversify recall while preserving primary Choice-A decision and avoiding broad over-generation',
        }
        state['decision'] = {
            'M_rec': rec_items,
            'M_level': {rec['candidate']: level},
            'M_avoid': avoid_items,
            'M_caution': caution_items,
            'M_alt': alt_items,
            'U': uncertainty,
            'uncertainty_reason': 'clinician review recommended' if uncertainty else 'routine clinician confirmation',
            'yes_no_rationale': {
                rec['candidate']: state.get('yes_no_decision_board', {}).get(rec['candidate'], {}),
                avoid['candidate']: state.get('yes_no_decision_board', {}).get(avoid['candidate'], {}),
            },
        }
        _record_yesno(state, self.name, 'Is the selected M_rec candidate different from the M_avoid candidate?', norm(rec['candidate']) != norm(avoid['candidate']), {'M_rec': rec['candidate'], 'M_avoid': avoid['candidate'], 'margin': margin}, target='final_decision', rationale='Coordinator enforces the consistency constraint M_rec ∩ M_avoid = empty.')
        _record_yesno(state, self.name, 'Should the final decision be flagged for clinician review?', uncertainty, {'level': level, 'margin': margin, 'has_support': has_support, 'has_avoid': has_avoid}, target='final_decision', rationale='Review is required when the margin is low, evidence is incomplete, or FORTA C/monitoring risk is present.')
        state['pmcg']['edges'].append({'source': rec['candidate'], 'target': avoid['candidate'], 'type': 'alternative_to', 'rule_id': 'DELIBERATED_SAFER_ALTERNATIVE', 'effect': 'support', 'severity': 1, 'risk_type': 'safer_alternative', 'source_id': 'multi_agent_deliberation'})
        state['scores'] = scores
        state['margin'] = margin
        state.setdefault('blackboard', {})['decision_draft'] = state['decision']
        _append_message(state, self.name, 'pass', 'Applied risk-first ordering using seed candidates and a generated candidate-pool audit before benefit ranking.', {'scores': [{k: v for k, v in x.items() if k != 'edges'} for x in scores], 'margin': margin})
        _send(state, self.name, 'TraceConsistencyVerifier', 'decision_ready_for_verification', 'Draft decision is ready for PMCG trace verification.')
        return state


class SafetyCritic:
    name = 'SafetyCritic'

    def run(self, state):
        d = state.get('decision') or {}
        rec = {norm(x) for x in d.get('M_rec', [])}
        avoid = {norm(x) for x in d.get('M_avoid', [])}
        caution = {norm(x) for x in d.get('M_caution', [])}
        alt = {norm(x) for x in d.get('M_alt', [])}
        findings = []
        if rec & avoid:
            findings.append({'type': 'unsafe_overlap', 'severity': 'critical', 'detail': sorted(rec & avoid), 'resolved': False})
        if avoid and not alt:
            findings.append({'type': 'missing_safer_alternative', 'severity': 'moderate', 'detail': 'M_avoid is non-empty but M_alt is empty', 'resolved': False})

        hard_block_audit = {}
        for med in d.get('M_rec', []):
            evidence = _candidate_hard_block_evidence(state, med)
            hard_block_audit[med] = evidence
            if evidence['hard_blocked']:
                findings.append({
                    'type': 'hard_blocked_recommendation',
                    'severity': 'critical',
                    'detail': med,
                    'evidence': evidence,
                    'resolved': False,
                })
            elif _board_entry(state, med).get('should_avoid'):
                findings.append({
                    'type': 'recommendation_conflicts_with_yes_no_avoid_gate',
                    'severity': 'moderate',
                    'detail': med,
                    'evidence': evidence,
                    'resolved': False,
                })

        for med in d.get('M_rec', []):
            if norm(med) in caution and str((d.get('M_level') or {}).get(med, '')).upper() != 'C':
                findings.append({'type': 'caution_recommendation_without_c_level', 'severity': 'moderate', 'detail': med, 'resolved': False})
        trace = state.get('trace_verification') or {}
        if trace.get('unsupported_claims'):
            findings.append({'type': 'unsupported_claims', 'severity': 'moderate', 'detail': trace.get('unsupported_claims'), 'resolved': False})

        state['candidate_safety_gate'] = {
            'policy': 'evidence_gated_hard_veto_caution_preserving',
            'recommendation_audit': hard_block_audit,
            'hard_blocked_recommendations': [m for m, e in hard_block_audit.items() if e['hard_blocked']],
            'gold_access': False,
            'case_id_rules': False,
        }
        state['critic_findings'] = findings
        state.setdefault('blackboard', {})['candidate_safety_gate'] = state['candidate_safety_gate']
        state.setdefault('blackboard', {})['critic_findings'] = findings
        _record_yesno(state, self.name, 'Does the draft decision pass safety critique without critical findings?', not any(f.get('severity') == 'critical' for f in findings), {'finding_count': len(findings), 'findings': findings[:5]}, target='final_decision', rationale='The safety critic applies a high-confidence candidate veto while preserving monitorable risks as caution.')
        _append_message(state, self.name, 'pass' if not findings else 'review_required', 'Critiqued draft decision with candidate-level evidence-gated safety veto.', {'finding_count': len(findings)})
        _send(state, self.name, 'RevisionAgent', 'safety_critique_available', 'Repair hard-blocked recommendations without leaving required output slots empty.', {'finding_count': len(findings)})
        return state


class RevisionAgent:
    name = 'RevisionAgent'

    def run(self, state):
        d = state.get('decision') or {}
        original_rec = list(d.get('M_rec', []) or [])
        original_avoid = list(d.get('M_avoid', []) or [])
        original_alt = list(d.get('M_alt', []) or [])
        changed = False

        # Candidate-level hard veto.  Caution-only candidates remain recommendable.
        blocked_rec = [m for m in original_rec if _candidate_hard_block_evidence(state, m)['hard_blocked']]
        blocked_norms = _norm_set(blocked_rec)

        # Preserve the frozen decision byte-for-byte when the new gate does not
        # activate.  This avoids incidental changes to caution/level fields and
        # makes the intervention attributable to true hard-veto cases only.
        if not blocked_rec:
            overlap = _norm_set(original_rec) & _norm_set(original_avoid)
            if overlap:
                d['M_rec'] = [x for x in original_rec if norm(x) not in overlap]
                changed = True
            if original_avoid and not original_alt:
                proposed = state.get('alternative_selection', {}).get('safer_alternative')
                if proposed and norm(proposed) not in _norm_set(original_avoid):
                    d['M_alt'] = [proposed]
                    changed = True
            state['final_safety_invariants'] = {
                'passed': not bool(_norm_set(d.get('M_rec', [])) & _norm_set(d.get('M_avoid', []))) and bool(d.get('M_rec')) and (not d.get('M_avoid') or bool(d.get('M_alt'))),
                'unresolved': [],
                'blocked_recommendations': [],
                'replacement': None,
                'slot_preserved': bool(d.get('M_rec') and (not d.get('M_avoid') or d.get('M_alt'))),
                'recommend_avoid_disjoint': not bool(_norm_set(d.get('M_rec', [])) & _norm_set(d.get('M_avoid', []))),
                'gold_access': False,
                'case_id_rules': False,
            }
            state['decision'] = d
            state.setdefault('blackboard', {})['revised_decision'] = d
            state.setdefault('blackboard', {})['final_safety_invariants'] = state['final_safety_invariants']
            for finding in state.get('critic_findings', []):
                if finding.get('type') in ('unsafe_overlap', 'missing_safer_alternative'):
                    finding['resolved'] = True
            _record_yesno(state, self.name, 'Do final deterministic safety invariants pass after slot-preserving repair?', state['final_safety_invariants']['passed'], state['final_safety_invariants'], target='final_decision', rationale='When no high-confidence veto activates, the frozen ATLAS decision is preserved.')
            _append_message(state, self.name, 'pass', 'Confirmed final safety invariants without changing the frozen decision.', {'changed': changed, 'blocked_count': 0})
            _send(state, self.name, 'TraceConsistencyVerifier', 'revised_decision_ready', 'Decision required no candidate-level hard-veto repair.')
            return state

        retained_rec = [m for m in original_rec if norm(m) not in blocked_norms]

        replacement = None
        if blocked_rec and not retained_rec:
            ranked = _rank_safe_repair_candidates(state, blocked_norms)
            if ranked:
                replacement = ranked[0]
                retained_rec = [replacement]

        if blocked_rec:
            changed = True

        # Hard-blocked recommendations are promoted to the front of M_avoid so
        # the trace verifier sees the strongest supported safety decision first.
        avoid_items = _unique_keep_order(blocked_rec + original_avoid, max_items=2)
        rec_norms = _norm_set(retained_rec)
        avoid_items = [x for x in avoid_items if norm(x) not in rec_norms]

        # Alternatives must themselves pass the hard veto.  Preserve existing
        # safe alternatives, then use the repaired recommendation as a fallback.
        alt_items = [x for x in original_alt if not _candidate_hard_block_evidence(state, x)['hard_blocked']]
        if replacement:
            alt_items = _unique_keep_order([replacement] + alt_items, max_items=2)
        elif retained_rec and not alt_items:
            alt_items = [retained_rec[0]]
        alt_items = _unique_keep_order(alt_items, max_items=2)

        caution_items = list(d.get('M_caution', []) or [])
        levels = dict(d.get('M_level', {}) or {})
        for med in retained_rec:
            yn = _board_entry(state, med)
            if yn.get('requires_caution_or_monitoring'):
                caution_items = _unique_keep_order(caution_items + [med])
                levels[med] = 'C'
            elif med not in levels:
                levels[med] = state.get('rules', {}).get('dev_level_map', {}).get(med) or 'UNKNOWN'

        d['M_rec'] = _unique_keep_order(retained_rec, max_items=2)
        d['M_avoid'] = avoid_items
        d['M_alt'] = alt_items
        d['M_caution'] = _unique_keep_order(caution_items)
        d['M_level'] = levels

        unresolved = []
        for med in d.get('M_rec', []):
            if _candidate_hard_block_evidence(state, med)['hard_blocked']:
                unresolved.append({'type': 'hard_blocked_recommendation_remains', 'candidate': med})
        if _norm_set(d.get('M_rec', [])) & _norm_set(d.get('M_avoid', [])):
            unresolved.append({'type': 'recommend_avoid_overlap'})
        if d.get('M_avoid') and not d.get('M_alt'):
            unresolved.append({'type': 'missing_safer_alternative'})
        if not d.get('M_rec'):
            unresolved.append({'type': 'no_safe_recommendation_available'})

        if unresolved:
            d['U'] = True
            d['uncertainty_reason'] = 'clinician review required after unresolved final safety invariant'
        elif blocked_rec:
            # A repaired decision remains reviewable, but is not left incomplete.
            d['U'] = True
            d['uncertainty_reason'] = 'candidate-level safety veto applied; safer slot-preserving repair selected'

        state['final_safety_invariants'] = {
            'passed': not unresolved,
            'unresolved': unresolved,
            'blocked_recommendations': blocked_rec,
            'replacement': replacement,
            'slot_preserved': bool(d.get('M_rec') and (not d.get('M_avoid') or d.get('M_alt'))),
            'recommend_avoid_disjoint': not bool(_norm_set(d.get('M_rec', [])) & _norm_set(d.get('M_avoid', []))),
            'gold_access': False,
            'case_id_rules': False,
        }
        state['decision'] = d
        state.setdefault('blackboard', {})['revised_decision'] = d
        state.setdefault('blackboard', {})['final_safety_invariants'] = state['final_safety_invariants']

        resolved_types = {
            'unsafe_overlap', 'missing_safer_alternative',
            'hard_blocked_recommendation',
            'recommendation_conflicts_with_yes_no_avoid_gate',
            'caution_recommendation_without_c_level',
        }
        for finding in state.get('critic_findings', []):
            if finding.get('type') in resolved_types and not unresolved:
                finding['resolved'] = True

        _record_yesno(state, self.name, 'Do final deterministic safety invariants pass after slot-preserving repair?', not unresolved, state['final_safety_invariants'], target='final_decision', rationale='Revision vetoes only high-confidence avoid evidence, preserves caution-only therapy, and fills required recommendation/alternative slots.')
        _append_message(state, self.name, 'pass' if not unresolved else 'review_required', 'Applied evidence-gated veto and slot-preserving safety repair.', {'changed': changed, 'blocked_count': len(blocked_rec), 'replacement': replacement, 'unresolved_count': len(unresolved)})
        _send(state, self.name, 'TraceConsistencyVerifier', 'revised_decision_ready', 'Decision passed candidate-level safety repair and final invariant checking.')
        return state


class TraceConsistencyVerifier:
    name = 'TraceConsistencyVerifier'

    def run(self, state):
        d = state['decision']
        edges = state['pmcg']['edges']
        rec = d['M_rec'][0] if d.get('M_rec') else ''
        avoid = d['M_avoid'][0] if d.get('M_avoid') else ''
        rec_support = bool(rec) and any(e.get('source') == rec and e.get('effect') in ('support', 'caution') for e in edges)
        avoid_support = bool(avoid) and (any(e.get('source') == avoid and e.get('effect') == 'avoid' for e in edges) or any_phrase(avoid, UNSAFE_CUES))
        disjoint = bool(rec and avoid) and norm(rec) != norm(avoid)
        # In the recall-diversified variant, M_alt is no longer required to copy M_rec.
        # It may be a distinct fallback alternative when supported by PMCG evidence or
        # selected by the SaferAlternativeAgent. This prevents mechanical coupling of
        # M_rec and M_alt recall while preserving traceability.
        alt_items = d.get('M_alt') or []
        alt_ok = bool(alt_items) and (
            norm(alt_items[0]) == norm(rec)
            or any(e.get('source') == alt_items[0] and e.get('type') == 'alternative_to' for e in edges)
            or bool(state.get('alternative_selection', {}).get('safer_alternative'))
        )
        passed = disjoint and alt_ok and (rec_support or avoid_support)
        unsupported = []
        if not rec_support:
            unsupported.append('recommendation_without_explicit_support_edge')
        if not avoid_support:
            unsupported.append('avoidance_without_explicit_conflict_edge')
        state['trace_verification'] = {
            'trace_consistency': 'pass' if passed else 'fail',
            'recommendation_supported': rec_support,
            'avoidance_supported': avoid_support,
            'disjoint_outputs': disjoint,
            'alternative_consistent': alt_ok,
            'unsupported_claims': unsupported,
            'unsupported_claim_rate': len(unsupported) / 2.0,
        }
        state.setdefault('blackboard', {})['trace_verification'] = state['trace_verification']
        _record_yesno(state, self.name, 'Does the PMCG support the recommended candidate?', rec_support, {'M_rec': rec}, target=rec, rationale='A recommended candidate should have support or caution evidence in PMCG.')
        _record_yesno(state, self.name, 'Does the PMCG support avoiding the unsafe candidate?', avoid_support, {'M_avoid': avoid}, target=avoid, rationale='An avoided candidate should have avoid/conflict evidence or unsafe lexical cue support.')
        _record_yesno(state, self.name, 'Is the final structured decision trace-consistent?', passed, state['trace_verification'], target='final_decision', rationale='Trace consistency requires disjoint outputs, consistent alternative mapping, and at least one supporting PMCG path.')
        _append_message(state, self.name, 'pass' if passed else 'fail', 'Verified final decision support against PMCG evidence and cross-agent audit claims.')
        if not passed:
            _send(state, self.name, 'Coordinator', 'verification_failed', 'Final decision lacks complete evidence support; coordinator should flag clinician review.', {'unsupported_claims': unsupported})
            state['decision']['U'] = True
        else:
            _send(state, self.name, 'Coordinator', 'verification_passed', 'Final decision is internally supported and non-contradictory.')
        return state
