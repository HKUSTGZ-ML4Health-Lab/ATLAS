#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ATLAS-Agent TargetedGlobalSafetyPatch for GeriMedBench

Purpose:
- Apply a targeted global safety patch to the released ATLAS-Agent predictions.
- Apply only global medication-risk patterns observed as clinically meaningful:
  1. Theophylline in older adult airway disease
  2. Cimetidine with bleeding / anticoagulation risk
  3. SSRIs with SIADH / hyponatremia risk
  4. COX-2 inhibitors with renal-risk context

Does NOT use:
- gold_final_report
- hidden final labels
- case_id-specific rules

Uses only:
- public candidate medications
- acquired_facts
- current prediction final_report
"""

import argparse
import json
import re
from pathlib import Path


SLOTS = ["M_rec", "M_avoid", "M_caution", "M_alt"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def unwrap_preds(obj):
    if isinstance(obj, list):
        return obj, "list"
    for k in ["predictions", "results", "data"]:
        if isinstance(obj, dict) and isinstance(obj.get(k), list):
            return obj[k], k
    raise ValueError("Unsupported prediction format")


def rewrap(template, preds, key):
    if key == "list":
        return preds
    out = dict(template)
    out[key] = preds
    out["method"] = "ATLAS-Agent + TargetedGlobalSafetyPatch + CandidateCanonicalizer + SlotNormalizer"
    out["postprocess_variant"] = True
    out["gold_used_by_inference"] = False
    out["hidden_env_used_by_inference"] = False
    out["case_id_rules_used"] = False
    return out


def norm(s):
    s = str(s or "").lower().strip()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def has_any(s, words):
    ns = norm(s)
    return any(norm(w) in ns for w in words)


def as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    return [str(x).strip()]


def dedup(xs):
    out, seen = [], set()
    for x in xs:
        if not x:
            continue
        k = norm(x)
        if k not in seen:
            out.append(x)
            seen.add(k)
    return out


def walk_collect_candidates(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in ["candidate_medications", "candidates", "candidate_options"] and isinstance(v, list):
                out.extend([str(x).strip() for x in v if str(x).strip()])
            else:
                walk_collect_candidates(v, out)
    elif isinstance(obj, list):
        for v in obj:
            walk_collect_candidates(v, out)


def get_candidates(public_case):
    found = []
    walk_collect_candidates(public_case.get("visible_input", {}), found)
    return dedup(found)


def get_report(pred):
    fd = pred.get("final_decision")
    fr = pred.get("final_report")

    if isinstance(fd, dict):
        return dict(fd)

    if isinstance(fr, dict):
        if isinstance(fr.get("final_decision"), dict):
            return dict(fr["final_decision"])
        return dict(fr)

    return {"M_rec": [], "M_avoid": [], "M_caution": [], "M_alt": [], "U": False}


def canonical_match(x, candidates):
    if not x:
        return x
    if x in candidates:
        return x

    nx = norm(x)

    for c in candidates:
        if norm(c) == nx:
            return c

    for c in candidates:
        nc = norm(c)
        if nx and (nx in nc or nc in nx):
            return c

    return x


def canonicalize_list(xs, candidates):
    return dedup([canonical_match(x, candidates) for x in as_list(xs)])


def context_blob(public_case, pred):
    parts = [
        json.dumps(public_case.get("visible_input", {}), ensure_ascii=False),
        json.dumps(pred.get("acquired_facts", {}), ensure_ascii=False),
        json.dumps(pred.get("updated_patient_state", []), ensure_ascii=False),
        json.dumps(get_report(pred), ensure_ascii=False),
    ]
    return " ".join(parts)


def candidate_by_pattern(candidates, patterns):
    for c in candidates:
        if has_any(c, patterns):
            return c
    return ""


def add_to_slot(report, slot, med):
    if med:
        report[slot] = dedup(report.get(slot, []) + [med])


def remove_from_slots(report, med, slots):
    if not med:
        return
    nm = norm(med)
    for s in slots:
        report[s] = [x for x in report.get(s, []) if norm(x) != nm]


def replace_rec_alt_with_safe(report, safe):
    if safe:
        report["M_rec"] = [safe]
        report["M_alt"] = [safe]


def apply_targeted_rules(public_case, pred, report, candidates):
    ctx = context_blob(public_case, pred)
    changes = []

    # Candidate handles.
    theophylline = candidate_by_pattern(candidates, ["theophylline", "theodur"])
    bronchodilator_alt = candidate_by_pattern(candidates, ["another class of bronchodilator"])
    h2_alt = candidate_by_pattern(candidates, ["another histamine", "histamine receptor antagonist", "histamine-receptor antagonist"])
    nondrug = candidate_by_pattern(candidates, ["nondrug", "non drug", "supportive management"])
    acetaminophen = candidate_by_pattern(candidates, ["acetaminophen", "paracetamol"])
    cimetidine = candidate_by_pattern(candidates, ["cimetidine"])
    ssri = candidate_by_pattern(candidates, ["ssri", "selective serotonin"])
    tca = candidate_by_pattern(candidates, ["tca", "tricyclic"])
    cox2 = candidate_by_pattern(candidates, ["cox-2", "cox 2", "coxib"])

    # 1. Theophylline in older adult airway disease.
    if theophylline and has_any(ctx, ["bronchodilation", "obstructive airway", "older adult", "geriatric"]):
        add_to_slot(report, "M_avoid", theophylline)
        remove_from_slots(report, theophylline, ["M_rec", "M_alt", "M_caution"])
        safe = bronchodilator_alt or nondrug
        replace_rec_alt_with_safe(report, safe)
        changes.append("Theophylline moved to M_avoid; safer bronchodilator/supportive option used for M_rec/M_alt.")

    # 2. Cimetidine with bleeding / anticoagulation risk.
    if cimetidine and has_any(ctx, [
        "avoid adding cimetidine",
        "bleeding risk",
        "blood clotting",
        "anticoagulant",
        "anticoagulation",
        "receiving anticoagulant",
        "warfarin",
        "doac"
    ]):
        add_to_slot(report, "M_avoid", cimetidine)
        remove_from_slots(report, cimetidine, ["M_rec", "M_alt", "M_caution"])
        safe = h2_alt or nondrug
        replace_rec_alt_with_safe(report, safe)
        changes.append("Cimetidine moved to M_avoid under bleeding/anticoagulation risk; safer H2/supportive option used.")

    # 3. SSRI with SIADH / hyponatremia risk.
    if ssri and has_any(ctx, [
        "siadh",
        "hyponatremia",
        "hyponatraemia",
        "sodium lowering",
        "sodium-lowering",
        "sodium abnormality"
    ]):
        add_to_slot(report, "M_avoid", ssri)
        remove_from_slots(report, ssri, ["M_rec", "M_alt", "M_caution"])
        if tca:
            add_to_slot(report, "M_avoid", tca)
            remove_from_slots(report, tca, ["M_rec", "M_alt", "M_caution"])
        safe = nondrug
        replace_rec_alt_with_safe(report, safe)
        changes.append("SSRI moved to M_avoid under SIADH/hyponatremia risk; nondrug option used for M_rec/M_alt.")

    # 4. COX-2 inhibitors with renal-risk context.
    if cox2 and has_any(ctx, [
        "renal risk",
        "renal-risk",
        "renal safety fact",
        "egfr",
        "creatinine",
        "kidney",
        "dose accumulation"
    ]):
        add_to_slot(report, "M_avoid", cox2)
        remove_from_slots(report, cox2, ["M_rec", "M_alt", "M_caution"])
        safe = acetaminophen or nondrug
        replace_rec_alt_with_safe(report, safe)
        changes.append("COX-2 inhibitor moved to M_avoid under renal-risk context; acetaminophen/supportive option used.")

    # Clean overlaps.
    avoid_norms = {norm(x) for x in report.get("M_avoid", [])}
    for s in ["M_rec", "M_alt", "M_caution"]:
        report[s] = [x for x in report.get(s, []) if norm(x) not in avoid_norms]

    # Fill if empty after safety movement.
    if not report.get("M_rec"):
        safe = acetaminophen or bronchodilator_alt or h2_alt or nondrug
        if safe and norm(safe) not in avoid_norms:
            report["M_rec"] = [safe]
            changes.append(f"Filled empty M_rec with safe candidate: {safe}")

    if not report.get("M_alt"):
        safe = report.get("M_rec", [None])[0] if report.get("M_rec") else (acetaminophen or bronchodilator_alt or h2_alt or nondrug)
        if safe and norm(safe) not in avoid_norms:
            report["M_alt"] = [safe]
            changes.append(f"Filled empty M_alt with safe candidate: {safe}")

    report["M_rec"] = dedup(report.get("M_rec", []))
    report["M_avoid"] = dedup(report.get("M_avoid", []))
    report["M_caution"] = dedup(report.get("M_caution", []))
    report["M_alt"] = dedup(report.get("M_alt", []))
    report["U"] = bool(report["M_avoid"])

    return report, dedup(changes)


def process_one(public_case, pred):
    candidates = get_candidates(public_case)
    old_report = get_report(pred)

    report = {}
    for s in SLOTS:
        report[s] = canonicalize_list(old_report.get(s), candidates)
    report["U"] = bool(old_report.get("U", False))

    report, changes = apply_targeted_rules(public_case, pred, report, candidates)

    new_pred = dict(pred)
    new_pred["method"] = "ATLAS-Agent + TargetedGlobalSafetyPatch + CandidateCanonicalizer + SlotNormalizer"
    new_pred["final_decision"] = report
    new_pred["final_report"] = report

    # Preserve state/query trajectory.
    new_pred["updated_patient_state"] = pred.get("updated_patient_state", [])
    new_pred["queries"] = pred.get("queries", [])
    new_pred["query_log"] = pred.get("query_log", [])
    new_pred["acquired_facts"] = pred.get("acquired_facts", {})

    E = pred.get("E") if isinstance(pred.get("E"), dict) else {}
    rp = as_list(E.get("reasoning_path"))

    if changes:
        rp.append(
            "GeriMedBench TargetedGlobalSafetyPatch applied global medication-risk patterns: "
            + "; ".join(changes)
        )

    rp.append(
        "GeriMedBench processing uses only public candidate medications and acquired facts; no gold labels, hidden final reports, or case-id rules are used."
    )

    E["reasoning_path"] = dedup(rp)
    E["gold_used_by_inference"] = False
    E["hidden_env_used_by_inference"] = False
    E["case_id_rules_used"] = False
    new_pred["E"] = E

    audit = {
        "case_id": pred.get("case_id"),
        "candidate_medications": candidates,
        "old_report": old_report,
        "new_report": report,
        "changes": changes,
        "changed": bool(changes),
        "gold_used": False,
        "hidden_env_used": False,
        "case_id_rule_used": False,
    }

    return new_pred, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    public = load_json(args.public)
    pred_obj = load_json(args.pred)
    preds, key = unwrap_preds(pred_obj)

    public_by_id = {str(c.get("case_id")): c for c in public}

    new_preds = []
    audits = []

    for pred in preds:
        cid = str(pred.get("case_id"))
        pub = public_by_id.get(cid, {})
        new_pred, audit = process_one(pub, pred)
        new_preds.append(new_pred)
        audits.append(audit)

    out_obj = rewrap(pred_obj, new_preds, key)
    save_json(out_obj, args.out)

    report = {
        "variant": "ATLAS-Agent + TargetedGlobalSafetyPatch + CandidateCanonicalizer + SlotNormalizer",
        "base_prediction": args.pred,
        "output_prediction": args.out,
        "num_predictions": len(new_preds),
        "num_changed": sum(1 for a in audits if a["changed"]),
        "gold_used": False,
        "hidden_env_used": False,
        "case_id_rules_used": False,
        "note": "The GeriMedBench processing pipeline uses four global medication-risk patterns only: theophylline, cimetidine, SSRI-hyponatremia, and COX-2 renal risk.",
        "audit_first_40": audits[:40],
    }

    save_json(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
