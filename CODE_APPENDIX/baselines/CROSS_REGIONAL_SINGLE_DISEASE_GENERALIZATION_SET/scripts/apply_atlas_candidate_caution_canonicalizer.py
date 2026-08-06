#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATLAS + CandidateCautionCanonicalizer for ATLAS_SINGLE_DISEASE_GENERALIZATION.

Purpose
- Fix the previous free-text M_caution normalization error.
- This baseline test's M_caution is candidate-based: it should contain a candidate medication/class,
  not a free-text monitoring phrase.
- Modify ONLY final_decision.M_caution.
- Do NOT modify M_rec, M_avoid, M_alt, U, trace, or evidence.
- Do NOT read gold/reference.
- Do NOT use case_id rules.

Inputs
--input: baseline test_data/ATLAS_SINGLE_DISEASE_GENERALIZATION_input.json
--pred:  baselines_12/outputs/final_predictions_atlas.json
--out:   baselines_12/outputs/final_predictions_atlas_candidate_cautionnorm.json
--report baselines_12/results/candidate_cautionnorm_report.json
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_text(x: Any) -> str:
    s = str(x or "").lower().strip()
    s = re.sub(r"[^a-z0-9+]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    return [str(x).strip()] if str(x).strip() else []


def unwrap_predictions(obj: Any) -> Tuple[List[Dict[str, Any]], str]:
    if isinstance(obj, list):
        return obj, "list"
    if isinstance(obj, dict):
        for key in ["predictions", "results", "data"]:
            if isinstance(obj.get(key), list):
                return obj[key], key
    raise ValueError("Unsupported prediction format: expected list or dict with predictions/results/data.")


def rewrap_predictions(original: Any, preds: List[Dict[str, Any]], mode: str) -> Any:
    if mode == "list":
        return preds
    out = dict(original)
    out[mode] = preds
    return out


def candidate_names(case: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in case.get("candidate_medications", []) or []:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]).strip())
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())

    final, seen = [], set()
    for x in out:
        k = norm_text(x)
        if k and k not in seen:
            seen.add(k)
            final.append(x)
    return final


def project_to_candidate(x: Any, candidates: List[str]) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    ns = norm_text(s)
    for c in candidates:
        if ns == norm_text(c):
            return c
    for c in candidates:
        nc = norm_text(c)
        if ns and nc and (ns in nc or nc in ns):
            return c
    return ""


def project_list(xs: Any, candidates: List[str]) -> List[str]:
    out, seen = [], set()
    for x in as_list(xs):
        c = project_to_candidate(x, candidates)
        k = norm_text(c)
        if c and k not in seen:
            seen.add(k)
            out.append(c)
    return out


def get_final_decision(pred: Dict[str, Any]) -> Dict[str, Any]:
    fd = pred.get("final_decision")
    if not isinstance(fd, dict):
        fd = {}
        pred["final_decision"] = fd
    return fd


def extract_candidate_evidence(case: Dict[str, Any], candidates: List[str]) -> Dict[str, str]:
    ctx = ((case.get("clinical_state") or {}).get("context") or {})
    ev = str(ctx.get("candidate_evidence") or "")
    out = {c: "" for c in candidates}

    # Expected format: "candidate: evidence || candidate: evidence"
    parts = [p.strip() for p in ev.split("||") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        left, right = p.split(":", 1)
        cand = project_to_candidate(left.strip(), candidates)
        if cand:
            out[cand] = right.strip()

    # Also include surrounding case text as weak evidence fallback.
    joined_bits = []
    cs = case.get("clinical_state") or {}
    ctx = cs.get("context") or {}
    for key in ["clinical_problem", "risk_condition", "treatment_need", "treatment_indication", "risk_types", "monitoring_focus", "case_narrative"]:
        if ctx.get(key):
            joined_bits.append(str(ctx[key]))
    for c in candidates:
        if not out.get(c):
            out[c] = " | ".join(joined_bits)
    return out


MONITOR_CUES = [
    "monitor", "monitoring", "follow up", "follow-up", "review", "check",
    "renal", "kidney", "egfr", "creatinine", "potassium", "electrolyte",
    "glucose", "hypogly", "tolerance", "gastrointestinal", "gi",
    "sedation", "cognition", "fall", "orthostatic", "blood pressure", "pulse", "ecg",
    "bleeding", "respiratory", "exacerbation", "seizure", "bowel", "hydration",
    "urinary", "dizziness", "motor", "hallucination", "intraocular",
]

HARD_AVOID_CUES = [
    "avoid", "higher risk", "high risk", "contraind", "without monitoring",
    "without gastroprotection", "long acting sulfonylurea", "prolonged hypogly",
    "anticholinergic", "benzodiazepine", "z-drug", "tricyclic", "typical antipsychotic",
    "metoclopramide", "glibenclamide", "nsaid", "long-term nsaid", "pethidine", "pentazocine",
]

BENEFIT_CUES = [
    "matches", "active indication", "lower risk", "beneficial", "prefer", "selected",
    "appropriate", "first line", "safer", "replacement", "control", "therapy",
]


def text_has_any(text: str, cues: List[str]) -> bool:
    low = text.lower()
    return any(c in low for c in cues)


def score_caution_candidate(
    cand: str,
    evidence: str,
    recs: List[str],
    alts: List[str],
    avoids: List[str],
) -> float:
    nc = norm_text(cand)
    avoid_norm = {norm_text(x) for x in avoids}
    if nc in avoid_norm:
        return -999.0

    ev = str(evidence or "")
    score = 0.0

    # Preserve the current ATLAS primary decision shape: M_caution usually belongs to the selected safe candidate.
    if nc in {norm_text(x) for x in recs}:
        score += 8.0
    if nc in {norm_text(x) for x in alts}:
        score += 4.0

    if text_has_any(ev, MONITOR_CUES):
        score += 5.0
    if text_has_any(ev, BENEFIT_CUES):
        score += 3.0
    if text_has_any(ev, HARD_AVOID_CUES):
        score -= 8.0

    # Mild preference for non-empty evidence segments.
    if ev.strip():
        score += 1.0

    return score


def choose_candidate_caution(case: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[List[str], str]:
    candidates = candidate_names(case)
    fd = get_final_decision(pred)

    raw_caution = project_list(fd.get("M_caution"), candidates)
    recs = project_list(fd.get("M_rec"), candidates)
    avoids = project_list(fd.get("M_avoid"), candidates)
    alts = project_list(fd.get("M_alt"), candidates)

    evidence = extract_candidate_evidence(case, candidates)
    avoid_norm = {norm_text(x) for x in avoids}

    # 1. If raw M_caution already contains a valid candidate also present in M_rec/M_alt, keep one canonical item.
    for c in raw_caution:
        if norm_text(c) not in avoid_norm and (norm_text(c) in {norm_text(x) for x in recs + alts}):
            return [c], "keep_raw_candidate_overlap_rec_alt"

    # 2. If raw M_caution is valid and evidence supports monitoring, keep it.
    for c in raw_caution:
        if norm_text(c) not in avoid_norm and text_has_any(evidence.get(c, ""), MONITOR_CUES):
            return [c], "keep_raw_candidate_monitoring_evidence"

    # 3. Score candidates from rec/alt first; do not modify M_rec/M_alt, only use them as evidence.
    pool = []
    seen = set()
    for c in recs + alts + candidates:
        k = norm_text(c)
        if k and k not in seen:
            seen.add(k)
            pool.append(c)

    scored = []
    for c in pool:
        s = score_caution_candidate(c, evidence.get(c, ""), recs, alts, avoids)
        scored.append((s, c))
    scored.sort(key=lambda t: t[0], reverse=True)

    if scored and scored[0][0] > -100:
        return [scored[0][1]], f"candidate_score:{scored[0][0]:.1f}"

    # 4. Fallback: keep raw if nothing can be safely inferred.
    if raw_caution:
        return [raw_caution[0]], "fallback_raw_candidate"
    return [], "fallback_empty"


def normalize_one(case: Dict[str, Any], pred: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    out = dict(pred)
    fd = get_final_decision(out)
    old = as_list(fd.get("M_caution"))
    new, rule = choose_candidate_caution(case, out)
    fd["M_caution"] = new
    out["final_decision"] = fd
    return out, {
        "case_id": case.get("case_id"),
        "old_M_caution": old,
        "new_M_caution": new,
        "changed": old != new,
        "rule": rule,
        "gold_used": False,
        "case_id_rule_used": False,
        "modified_slots": ["M_caution"] if old != new else [],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    cases = load_json(args.input)
    pred_obj = load_json(args.pred)
    preds, mode = unwrap_predictions(pred_obj)

    if not isinstance(cases, list):
        raise ValueError("Input JSON must be a list of cases.")

    case_by_id = {str(c.get("case_id")): c for c in cases}
    new_preds: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    missing = 0

    for pred in preds:
        cid = str(pred.get("case_id"))
        case = case_by_id.get(cid)
        if case is None:
            new_preds.append(pred)
            audit.append({"case_id": cid, "changed": False, "rule": "missing_input_keep_unchanged", "gold_used": False})
            missing += 1
            continue
        new_pred, row = normalize_one(case, pred)
        new_preds.append(new_pred)
        audit.append(row)

    out_obj = rewrap_predictions(pred_obj, new_preds, mode)
    save_json(out_obj, args.out)

    report = {
        "method": "ATLAS + CandidateCautionCanonicalizer",
        "description": "Candidate-based M_caution canonicalization for ATLAS_SINGLE_DISEASE_GENERALIZATION.",
        "gold_used_by_aligner": False,
        "case_id_rules_used": False,
        "modified_slots": ["M_caution"],
        "unmodified_slots": ["M_rec", "M_avoid", "M_alt", "U", "trace"],
        "num_inputs": len(cases),
        "num_predictions": len(preds),
        "missing_input_for_predictions": missing,
        "num_changed_M_caution": sum(1 for r in audit if r.get("changed")),
        "audit_first_100": audit[:100],
    }
    save_json(report, args.report)

    print("[OK] wrote:", args.out)
    print("[OK] report:", args.report)
    print("[OK] gold used by aligner: False")
    print("[OK] modified slot: M_caution only")
    print("[OK] changed M_caution:", report["num_changed_M_caution"], "/", len(preds))


if __name__ == "__main__":
    main()
