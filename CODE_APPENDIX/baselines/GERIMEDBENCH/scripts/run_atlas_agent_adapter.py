#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the real ATLAS decision engine inside GeriMedBench-Asia-76.

Inference contract:
- Reads public cases.
- Reads hidden_env only through explicit env.query(query_type) calls.
- Does not read gold_final_report for inference.
- Does not use case-id-specific rules.
- Uses ATLAS UnifiedOrchestratorAgent for the final medication decision.
"""
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, List


def load_json(p: str) -> Any:
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save_json(obj: Any, p: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def norm(s: Any) -> str:
    return re.sub(r'\s+', ' ', str(s or '').lower()).strip()


def visible_text(case: Dict[str, Any]) -> str:
    return norm(json.dumps(case.get('visible_input') or {}, ensure_ascii=False))


def get_allowed_query_types(case: Dict[str, Any]) -> List[str]:
    out = []
    for q in case.get('allowed_queries', []) or []:
        qt = q.get('query_type') if isinstance(q, dict) else q
        if qt:
            out.append(str(qt))
    return out


def select_atlas_queries(public_case: Dict[str, Any], budget: int) -> List[str]:
    """Global public-only query policy for ATLAS-Agent.

    It uses public visible_input + allowed query names/descriptions only.
    It does not read hidden critical_facts, gold, or case_id.
    """
    allowed_items = public_case.get('allowed_queries', []) or []
    allowed, desc = [], {}
    for item in allowed_items:
        if isinstance(item, dict):
            q = item.get('query_type')
            d = item.get('description', '')
        else:
            q, d = item, ''
        if q:
            q = str(q)
            allowed.append(q)
            desc[q] = str(d)
    scores = {q: 0.0 for q in allowed}
    text = visible_text(public_case)

    def add(q, v):
        if q in scores:
            scores[q] += v
    def cue(words):
        return any(w in text for w in words)

    # Mild default geriatric priorities.
    add('renal_function', 0.8)
    add('drug_allergy', 0.4)

    if cue(['nsaid', 'ibuprofen', 'naproxen', 'diclofenac', 'aspirin', 'ulcer', 'gastro', 'gi bleeding', 'peptic', 'gastric']):
        add('gi_history', 4.0); add('renal_function', 2.5); add('bleeding_anticoagulation_status', 1.5)
    if cue(['warfarin', 'doac', 'anticoag', 'antiplatelet', 'clopidogrel', 'bleeding', 'asa', 'acetylsalicylic']):
        add('bleeding_anticoagulation_status', 4.0); add('gi_history', 2.0); add('renal_function', 1.5)
    if cue(['benzodiazepine', 'zolpidem', 'z-drug', 'sedat', 'opioid', 'tramadol', 'fall', 'fracture', 'syncope', 'dizziness']):
        add('fall_syncope_history', 4.0); add('cognitive_status', 2.5); add('sleep_history', 2.0); add('respiratory_status', 1.0)
    if cue(['anticholinergic', 'oxybutynin', 'amitriptyline', 'paroxetine', 'delirium', 'dementia', 'cognitive', 'confusion', 'tca', 'tricyclic']):
        add('cognitive_status', 4.0); add('urinary_symptom_history', 2.0); add('ophthalmic_glaucoma_history', 1.5); add('fall_syncope_history', 1.5)
    if cue(['urinary', 'retention', 'bph', 'incontinence', 'overactive bladder', 'oxybutynin']):
        add('urinary_symptom_history', 4.0); add('cognitive_status', 1.5)
    if cue(['insomnia', 'sleep', 'hypnotic', 'zolpidem', 'benzodiazepine']):
        add('sleep_history', 4.0); add('fall_syncope_history', 2.0); add('cognitive_status', 1.5)
    if cue(['parkinson', 'extrapyramidal', 'eps', 'metoclopramide', 'haloperidol', 'antipsychotic', 'chlorpromazine']):
        add('parkinsonism_history', 4.0); add('cognitive_status', 1.5); add('fall_syncope_history', 1.5); add('blood_pressure_status', 1.5)
    if cue(['copd', 'asthma', 'respiratory', 'bronchospasm', 'exacerbation', 'opioid', 'benzodiazepine']):
        add('respiratory_status', 4.0); add('fall_syncope_history', 1.0)
    if cue(['glaucoma', 'intraocular', 'narrow angle', 'anticholinergic']):
        add('ophthalmic_glaucoma_history', 4.0); add('cognitive_status', 1.0)
    if cue(['constipation', 'bowel', 'opioid', 'anticholinergic']):
        add('bowel_history', 4.0); add('urinary_symptom_history', 1.5)
    if cue(['heart failure', 'fluid', 'edema', 'oedema', 'pioglitazone', 'thiazolidinedione', 'nsaid', 'calcium channel blocker']):
        add('heart_failure_status', 4.0); add('renal_function', 1.5); add('blood_pressure_status', 1.0)
    if cue(['hypertension', 'blood pressure', 'orthostatic', 'hypotension', 'alpha blocker', 'prazosin', 'doxazosin', 'syncope']):
        add('blood_pressure_status', 4.0); add('fall_syncope_history', 2.0); add('renal_function', 1.0)
    if cue(['arrhythmia', 'atrial fibrillation', 'af ', 'ecg', 'qt', 'amiodarone', 'digoxin', 'pulse']):
        add('arrhythmia_history_ecg', 4.0); add('renal_function', 1.5); add('bleeding_anticoagulation_status', 1.0)
    if cue(['diabetes', 'hypogly', 'glucose', 'glibenclamide', 'glyburide', 'sulfonylurea', 'insulin', 'metformin']):
        add('diabetes_control', 4.0); add('renal_function', 2.5); add('fall_syncope_history', 1.0)
    if cue(['hyponatraemia', 'hyponatremia', 'sodium', 'siadh', 'ssri', 'diuretic', 'thiazide']):
        add('sodium_history', 4.0); add('renal_function', 1.5); add('fall_syncope_history', 1.0)
    if cue(['raynaud', 'vasospasm', 'beta blocker', 'propranolol']):
        add('raynaud_symptoms', 4.0); add('blood_pressure_status', 1.0)

    # Weak signal from query descriptions.
    for q, d in desc.items():
        for tok in re.split(r'[^a-zA-Z0-9]+', d.lower()):
            if len(tok) >= 6 and tok in text:
                add(q, 0.5)
                break

    ranked = sorted(allowed, key=lambda q: (-scores.get(q, 0.0), allowed.index(q)))
    selected = [q for q in ranked if scores.get(q, 0.0) > 0][:budget]
    return selected or allowed[:budget]


class HiddenEnv:
    def __init__(self, hidden_cases: List[Dict[str, Any]]):
        self.by_id = {c.get('case_id'): c for c in hidden_cases}
    def query(self, case_id: str, query_type: str) -> Any:
        c = self.by_id.get(case_id) or {}
        return (c.get('hidden_facts') or {}).get(query_type, 'not available')


def get_candidates(visible: Dict[str, Any]) -> List[str]:
    vals = visible.get('candidate_medications') or visible.get('candidates') or []
    return [str(x.get('name') if isinstance(x, dict) else x) for x in vals if str(x).strip()]


def acquired_state_updates(acquired: Dict[str, Any]) -> List[str]:
    updates = []
    for k, v in acquired.items():
        txt = norm(v)
        label = k.replace('_', ' ')
        if 'no additional' in txt or 'no known' in txt or txt == 'not available':
            updates.append(f'{label} reviewed: no additional risk documented')
        else:
            updates.append(f'{label} acquired: {v}')
    return updates


def build_atlas_case(public_case: Dict[str, Any], acquired: Dict[str, Any]) -> Dict[str, Any]:
    visible = public_case.get('visible_input') or {}
    patient = visible.get('patient') or {}
    candidates = get_candidates(visible)
    therapeutic_need = visible.get('therapeutic_need', '')
    under_review = visible.get('initial_medication_under_review', '')
    known_context = visible.get('known_context', []) or []
    acquired_lines = [f'{k}: {v}' for k, v in acquired.items()]

    age = patient.get('age') or patient.get('age_min') or 75
    try:
        age_min = int(age)
    except Exception:
        age_min = 75

    context = {
        'clinical_problem': therapeutic_need or 'geriatric medication safety review',
        'treatment_need': therapeutic_need,
        'treatment_indication': therapeutic_need,
        'initial_medication_under_review': under_review,
        'known_context': ' || '.join(map(str, known_context)),
        'acquired_facts': ' || '.join(acquired_lines),
        'candidate_evidence': ' || '.join([f'Candidate option: {c}' for c in candidates]),
        'candidate_scope': 'Closed candidate set: use only candidate_medications for medication, class, or non-pharmacological candidate decisions.',
        'decision_contract': (
            'Produce candidate-constrained medication safety output. M_rec, M_avoid, M_caution and M_alt must use candidate names from candidate_medications when possible. '
            'M_caution should contain a candidate requiring monitoring, not free-text monitoring instructions. Avoid high-risk option if acquired facts establish safety conflict.'
        ),
        'interactive_agent_benchmark': 'GeriMedBench-Asia-76',
        'gold_visibility': 'not available to inference',
    }
    clinical_state = {
        'diseases': [str(x) for x in [therapeutic_need, under_review, 'Asian older adult geriatric medication safety'] if str(x).strip()],
        'current_medications': [under_review] if under_review else [],
        'age_related_factors': ['older_age', 'asian_geriatric_population'],
        'risk_factors': [str(x) for x in acquired_lines] + [str(x) for x in known_context],
        'context': context,
    }
    return {
        'case_id': public_case.get('case_id'),
        'patient': {
            'age_group': 'older_adult',
            'age_min': age_min,
            'sex': patient.get('sex', 'unknown'),
            'region': patient.get('region', 'Asia'),
        },
        'clinical_state': clinical_state,
        'candidate_medications': [{'name': c, 'entity_type': 'medication_or_class'} for c in candidates],
        'task': 'gerimedbench_asia76_interactive_atlas_agent_candidate_constrained_medication_safety_reasoning',
        'input_safety': {
            'contains_gold_labels': False,
            'contains_hidden_answers_only_after_explicit_query': True,
            'case_id_rule_used': False,
        }
    }


def load_atlas(atlas_root: Path):
    sys.path.insert(0, str(atlas_root))
    from framework_src.agents.unified_orchestrator import UnifiedOrchestratorAgent  # noqa
    rules_path = atlas_root / '02_FROZEN_INFERENCE' / 'frozen' / 'frozen_rules.json'
    rules = load_json(str(rules_path))
    return UnifiedOrchestratorAgent(rules)


def run_case(orchestrator, public_case: Dict[str, Any], env: HiddenEnv, budget: int, method: str) -> Dict[str, Any]:
    cid = public_case.get('case_id')
    queries = select_atlas_queries(public_case, budget)
    acquired = {q: env.query(cid, q) for q in queries}
    query_log = [{'query_type': q, 'answer': acquired[q], 'source': 'hidden_env_via_explicit_query'} for q in queries]
    atlas_case = build_atlas_case(public_case, acquired)
    atlas_out = orchestrator.run_case(atlas_case)
    final_decision = atlas_out.get('final_decision') or {'M_rec': [], 'M_avoid': [], 'M_caution': [], 'M_alt': [], 'M_level': {}, 'U': False}
    # Preserve ATLAS decision unchanged. Only wrap schema for evaluator compatibility.
    reasoning = []
    for q, ans in acquired.items():
        reasoning.append(f'Acquired {q}: {ans}')
    reasoning += [str(x) for x in atlas_out.get('reasoning_trace', [])]
    if not reasoning:
        reasoning = ['ATLAS produced final decision from acquired facts and visible patient state.']
    return {
        'case_id': cid,
        'method': method,
        'queries': queries,
        'query_log': query_log,
        'acquired_facts': acquired,
        'updated_patient_state': acquired_state_updates(acquired),
        'final_decision': final_decision,
        'final_report': final_decision,
        'atlas_case_input': atlas_case,
        'atlas_raw_output': atlas_out,
        'E': {
            'acquired_facts': list(acquired.keys()),
            'reasoning_path': reasoning,
            'gold_used_by_inference': False,
            'case_id_rules_used': False,
            'final_decision_engine': 'ATLAS UnifiedOrchestratorAgent',
        }
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas_root', default=None, help='Path to main ATLAS root. Default: parent of agent_benchmark directory.')
    ap.add_argument('--public', required=True)
    ap.add_argument('--hidden', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--budget', type=int, default=3)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--method', default='ATLAS-Agent')
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parents[1]
    atlas_root = Path(args.atlas_root).resolve() if args.atlas_root else default_root
    public = load_json(args.public)
    hidden = load_json(args.hidden)
    if args.limit and args.limit > 0:
        public = public[:args.limit]
    env = HiddenEnv(hidden)
    orchestrator = load_atlas(atlas_root)
    preds = []
    for i, pc in enumerate(public, 1):
        print(f'[{i}/{len(public)}] {args.method} {pc.get("case_id")}', flush=True)
        try:
            preds.append(run_case(orchestrator, pc, env, args.budget, args.method))
        except Exception as e:
            preds.append({
                'case_id': pc.get('case_id'),
                'method': args.method,
                'error': repr(e),
                'queries': [], 'query_log': [], 'acquired_facts': {}, 'updated_patient_state': [],
                'final_decision': {'M_rec': [], 'M_avoid': [], 'M_caution': [], 'M_alt': [], 'M_level': {}, 'U': False},
                'final_report': {'M_rec': [], 'M_avoid': [], 'M_caution': [], 'M_alt': [], 'M_level': {}, 'U': False},
                'E': {'reasoning_path': [f'ATLAS-Agent runner error: {repr(e)}'], 'gold_used_by_inference': False, 'case_id_rules_used': False}
            })
    result = {
        'benchmark': 'GeriMedBench-Asia-76-Agent',
        'method': args.method,
        'mode': 'atlas_main_system_adapter',
        'atlas_root': str(atlas_root),
        'query_budget': args.budget,
        'gold_used_by_inference': False,
        'hidden_env_used_only_via_explicit_query': True,
        'case_id_rules_used': False,
        'predictions': preds,
    }
    save_json(result, args.out)
    print(f'[OK] wrote {args.out}')

if __name__ == '__main__':
    main()
