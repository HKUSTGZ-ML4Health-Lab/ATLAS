#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed GeriMedBench-Asia agent evaluator.

Fixes compared with earlier scaffold evaluator:
- reads hidden.gold_final_report as the evaluator gold field;
- supports prediction.final_decision and prediction.final_report;
- keeps gold access evaluator-only;
- success is counted before error attribution labels.
"""
import argparse, json, re
from pathlib import Path
from statistics import mean

SLOTS = ["M_rec", "M_avoid", "M_caution", "M_alt"]

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def unwrap_preds(obj):
    if isinstance(obj, list): return obj
    for k in ["predictions", "results", "data"]:
        if isinstance(obj, dict) and isinstance(obj.get(k), list):
            return obj[k]
    raise ValueError('Unsupported prediction file format')

def norm(s):
    s = str(s or '').strip().lower()
    s = s.replace('–','-').replace('—','-')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def norm_set(xs):
    if xs is None: return set()
    if isinstance(xs, str): xs = [xs]
    return {norm(x) for x in xs if norm(x)}

def prf(pred_set, gold_set):
    if not pred_set and not gold_set: return 1.0, 1.0, 1.0
    if not pred_set and gold_set: return 0.0, 0.0, 0.0
    if pred_set and not gold_set: return 0.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else 0.0
    r = tp / len(gold_set) if gold_set else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1

def pct(x): return round(100.0 * x, 2)

def state_hit(pred_item, gold_item):
    p, g = norm(pred_item), norm(gold_item)
    if not p or not g: return False
    if g in p or p in g: return True
    p_tokens, g_tokens = set(p.split()), set(g.split())
    return len(p_tokens & g_tokens) >= max(1, min(2, len(g_tokens)))

def state_f1(pred_states, gold_states):
    pred, gold = list(pred_states or []), list(gold_states or [])
    if not pred and not gold: return 1.0
    if not pred or not gold: return 0.0
    matched_gold, tp = set(), 0
    for pi in pred:
        for j, gi in enumerate(gold):
            if j in matched_gold: continue
            if state_hit(pi, gi):
                matched_gold.add(j); tp += 1; break
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

def trace_consistent(pred, critical_facts):
    e = pred.get('E', {}) or {}
    acquired = set((pred.get('acquired_facts') or {}).keys()) | set(e.get('acquired_facts', []) or [])
    rp = ' '.join(e.get('reasoning_path', []) or pred.get('reasoning_path', []) or [])
    rp_low = rp.lower()
    crit_acq = [c for c in critical_facts if c in acquired]
    if not crit_acq: return False
    if len(rp_low.strip()) < 20: return False
    for c in crit_acq:
        if c.replace('_', ' ') in rp_low or c in rp_low:
            return True
    return any(c.split('_')[0] in rp_low for c in crit_acq)

def get_pred_report(pred):
    fd = pred.get('final_decision') or pred.get('final_report') or {}
    if isinstance(fd, dict) and isinstance(fd.get('final_decision'), dict):
        fd = fd['final_decision']
    return fd if isinstance(fd, dict) else {}

def get_gold_report(hidden):
    gd = hidden.get('gold_final_decision') or hidden.get('gold_final_report') or {}
    if isinstance(gd, dict) and isinstance(gd.get('final_decision'), dict):
        gd = gd['final_decision']
    return gd if isinstance(gd, dict) else {}

def evaluate_case(pred, hidden):
    critical = set(hidden.get('critical_facts', []))
    useful = set(hidden.get('useful_facts', [])) | critical
    queries = pred.get('queries', []) or []
    queries = [q.get('query_type') if isinstance(q, dict) else q for q in queries]
    queries = [str(q) for q in queries]
    unique_queries = list(dict.fromkeys(queries))
    acquired = set(unique_queries)
    info_gain = len(acquired & critical) / len(critical) if critical else 1.0
    useful_nonredundant = len(set(unique_queries) & useful)
    query_eff = useful_nonredundant / len(queries) if queries else 0.0
    rev_f1 = state_f1(pred.get('updated_patient_state', []), hidden.get('gold_state_updates', []))
    fd = get_pred_report(pred)
    gold_fd = get_gold_report(hidden)
    slot_scores, slot_exact = {}, True
    for s in SLOTS:
        ps, gs = norm_set(fd.get(s, [])), norm_set(gold_fd.get(s, []))
        p, r, f1 = prf(ps, gs)
        slot_scores[s] = {'precision': p, 'recall': r, 'f1': f1, 'exact': ps == gs}
        slot_exact = slot_exact and (ps == gs)
    pred_unsafe = bool(
        (norm_set(fd.get('M_rec', [])) & norm_set(gold_fd.get('M_avoid', []))) or
        (norm_set(fd.get('M_alt', [])) & norm_set(gold_fd.get('M_avoid', [])))
    )
    expected_u = bool(gold_fd.get('U', False))
    u_correct = bool(fd.get('U', False)) == expected_u
    trace_ok = trace_consistent(pred, critical)
    final_strict = slot_exact and (not pred_unsafe) and u_correct and trace_ok
    if final_strict:
        err = 'success'
    elif info_gain < 0.999:
        err = 'acquisition_error'
    elif rev_f1 < 0.999:
        err = 'state_revision_error'
    elif not trace_ok:
        err = 'trace_error'
    else:
        err = 'downstream_decision_error'
    final_slot_f1 = mean([slot_scores[s]['f1'] for s in SLOTS])
    osrs_agent = 0.25*info_gain + 0.15*query_eff + 0.20*rev_f1 + 0.30*final_slot_f1 + 0.10*(1.0 if trace_ok else 0.0)
    return {
        'case_id': pred.get('case_id'),
        'info_gain': info_gain,
        'query_efficiency': query_eff,
        'revision_accuracy': rev_f1,
        'final_strict': final_strict,
        'trace_consistency': trace_ok,
        'unsafe': pred_unsafe,
        'U_exact': u_correct,
        'slot_scores': slot_scores,
        'error_type': err,
        'agent_osrs': osrs_agent,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--public', required=True, help='kept for manifest; not used for gold')
    ap.add_argument('--hidden', required=True)
    ap.add_argument('--pred', required=True)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--detail', required=True)
    args = ap.parse_args()
    hidden_obj = load_json(args.hidden)
    hidden_cases = hidden_obj['cases'] if isinstance(hidden_obj, dict) and 'cases' in hidden_obj else hidden_obj
    hidden_by_id = {c['case_id']: c for c in hidden_cases}
    pred_obj = load_json(args.pred)
    preds = unwrap_preds(pred_obj)
    rows = []
    for p in preds:
        cid = p.get('case_id')
        if cid in hidden_by_id:
            rows.append(evaluate_case(p, hidden_by_id[cid]))
    N = len(rows)
    def avg(key): return mean([r[key] for r in rows]) if rows else 0.0
    err_counts = {}
    for r in rows:
        err_counts[r['error_type']] = err_counts.get(r['error_type'], 0) + 1
    summary = {
        'benchmark': 'GeriMedBench-Agent',
        'method': pred_obj.get('method', 'unknown') if isinstance(pred_obj, dict) else 'unknown',
        'N': N,
        'Info_Gain': pct(avg('info_gain')),
        'Query_Efficiency': pct(avg('query_efficiency')),
        'Revision_Accuracy': pct(avg('revision_accuracy')),
        'Final_Strict': pct(avg('final_strict')),
        'Unsafe_Rate': pct(avg('unsafe')),
        'Trace_Consistency': pct(avg('trace_consistency')),
        'Agent_OSRS': pct(avg('agent_osrs')),
        'Error_Attribution_Counts': err_counts,
        'gold_used_by_evaluator_only': True,
        'evaluator_fix': 'gold_final_report_schema_and_error_attribution_fixed',
    }
    save_json(summary, args.summary)
    save_json({'summary': summary, 'cases': rows}, args.detail)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == '__main__': main()
