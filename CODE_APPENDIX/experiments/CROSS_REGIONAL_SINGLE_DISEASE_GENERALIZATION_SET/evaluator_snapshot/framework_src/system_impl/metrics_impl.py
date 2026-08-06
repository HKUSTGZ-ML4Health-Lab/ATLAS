"""Detailed evaluation metrics for geriatric medication safety reasoning.

This evaluator reports both:
1) case-level hit metrics, for backward compatibility; and
2) item/set-level precision, recall, F1, and hit fractions for M_rec,
   M_avoid, M_caution, and M_alt.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set


def normalize_med_name(value: Any) -> str:
    """Normalize medication / label strings for approximate exact matching.

    This intentionally uses conservative normalization only: lowercasing,
    trimming, and removing punctuation. It does not use external medical synonym
    expansion, so evaluation remains auditable and deterministic.
    """
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_set(values: Any) -> Set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return set()
    output = set()
    for item in values:
        norm = normalize_med_name(item)
        if norm:
            output.add(norm)
    return output


def safe_divide(num: float, den: float, empty_value: float = 0.0) -> float:
    if den == 0:
        return empty_value
    return num / den


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _get_pred_decision(prediction: Dict[str, Any]) -> Dict[str, Any]:
    return prediction.get("final_decision", prediction)


def _get_gold_label(gold_row: Dict[str, Any]) -> Dict[str, Any]:
    return gold_row.get("gold_label", gold_row)


def set_scores(pred_items: Any, gold_items: Any) -> Dict[str, Any]:
    """Return per-case set-level scores and item lists."""
    pred = normalize_set(pred_items)
    gold = normalize_set(gold_items)

    tp = pred & gold
    fp = pred - gold
    fn = gold - pred

    precision = safe_divide(len(tp), len(pred), empty_value=1.0 if not gold else 0.0)
    recall = safe_divide(len(tp), len(gold), empty_value=1.0)
    f1 = f1_score(precision, recall)

    return {
        "gold_count": len(gold),
        "pred_count": len(pred),
        "hit_count": len(tp),
        "miss_count": len(fn),
        "extra_count": len(fp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_fraction": f"{len(tp)}/{len(gold)}" if gold else "NA",
        "hit_items": sorted(tp),
        "missed_items": sorted(fn),
        "extra_items": sorted(fp),
    }


def build_failure_reasons(
    rec_scores: Dict[str, Any],
    avoid_scores: Dict[str, Any],
    caution_scores: Dict[str, Any],
    alt_scores: Dict[str, Any],
    unsafe_items: List[str],
    trace_pass: bool,
) -> List[str]:
    reasons: List[str] = []

    if rec_scores["gold_count"] > 0 and rec_scores["recall"] < 1.0:
        reasons.append("incomplete_recommended_medication_recall")
    if avoid_scores["gold_count"] > 0 and avoid_scores["recall"] < 1.0:
        reasons.append("missed_avoid_medication")
    if caution_scores["gold_count"] > 0 and caution_scores["recall"] < 1.0:
        reasons.append("missed_caution_medication")
    if alt_scores["gold_count"] > 0 and alt_scores["recall"] < 1.0:
        reasons.append("missed_safer_alternative")
    if unsafe_items:
        reasons.append("unsafe_recommendation")
    if not trace_pass:
        reasons.append("trace_verification_failed")

    return reasons


def classify_case(
    rec_scores: Dict[str, Any],
    avoid_scores: Dict[str, Any],
    caution_scores: Dict[str, Any],
    alt_scores: Dict[str, Any],
    unsafe_items: List[str],
    trace_pass: bool,
) -> str:
    """Assign a mutually exclusive category for easier error analysis."""
    if unsafe_items:
        return "unsafe_failure"
    if not trace_pass:
        return "trace_failure"
    if avoid_scores["gold_count"] > 0 and avoid_scores["recall"] < 1.0:
        return "avoid_failure"
    if rec_scores["gold_count"] > 0 and rec_scores["recall"] < 1.0:
        return "recommendation_failure"
    if caution_scores["gold_count"] > 0 and caution_scores["recall"] < 1.0:
        return "caution_incomplete"
    if alt_scores["gold_count"] > 0 and alt_scores["recall"] < 1.0:
        return "alternative_incomplete"
    return "perfect_success"



def classify_case_simplified(
    rec_scores: Dict[str, Any],
    avoid_scores: Dict[str, Any],
    caution_scores: Dict[str, Any],
    alt_scores: Dict[str, Any],
    unsafe_items: List[str],
    trace_pass: bool,
) -> str:
    """Assign a publication-friendly mutually exclusive failure group.

    Priority order is intentionally safety-first:
    1. safety_critical_failure: unsafe recommendation or missed avoid medication
    2. evidence_failure: trace/evidence verification failed
    3. clinical_completeness_failure: recommended/caution/alternative outputs incomplete
    4. success: no simplified failure condition
    """
    if unsafe_items or (avoid_scores["gold_count"] > 0 and avoid_scores["recall"] < 1.0):
        return "safety_critical_failure"
    if not trace_pass:
        return "evidence_failure"
    if (
        (rec_scores["gold_count"] > 0 and rec_scores["recall"] < 1.0)
        or (caution_scores["gold_count"] > 0 and caution_scores["recall"] < 1.0)
        or (alt_scores["gold_count"] > 0 and alt_scores["recall"] < 1.0)
    ):
        return "clinical_completeness_failure"
    return "success"

def evaluate_case_detailed(pred_case: Dict[str, Any], gold_case: Dict[str, Any]) -> Dict[str, Any]:
    pred = _get_pred_decision(pred_case)
    gold = _get_gold_label(gold_case)

    rec_scores = set_scores(pred.get("M_rec", []), gold.get("M_rec", []))
    avoid_scores = set_scores(pred.get("M_avoid", []), gold.get("M_avoid", []))
    caution_scores = set_scores(pred.get("M_caution", []), gold.get("M_caution", []))
    alt_scores = set_scores(pred.get("M_alt", []), gold.get("M_alt", []))

    pred_rec = normalize_set(pred.get("M_rec", []))
    gold_avoid = normalize_set(gold.get("M_avoid", []))
    unsafe_items = sorted(pred_rec & gold_avoid)

    trace_info = pred_case.get("trace_verification", {}) or {}
    trace_pass = trace_info.get("trace_consistency") == "pass"
    unsupported_claim_rate = float(trace_info.get("unsupported_claim_rate", 0.0) or 0.0)

    failure_reasons = build_failure_reasons(
        rec_scores=rec_scores,
        avoid_scores=avoid_scores,
        caution_scores=caution_scores,
        alt_scores=alt_scores,
        unsafe_items=unsafe_items,
        trace_pass=trace_pass,
    )
    case_category = classify_case(
        rec_scores=rec_scores,
        avoid_scores=avoid_scores,
        caution_scores=caution_scores,
        alt_scores=alt_scores,
        unsafe_items=unsafe_items,
        trace_pass=trace_pass,
    )
    simplified_case_category = classify_case_simplified(
        rec_scores=rec_scores,
        avoid_scores=avoid_scores,
        caution_scores=caution_scores,
        alt_scores=alt_scores,
        unsafe_items=unsafe_items,
        trace_pass=trace_pass,
    )

    # Strict failure: any incomplete critical field, unsafe output, or trace failure.
    # This is intentionally stricter than the earlier hit-based failed-case rule.
    failed = bool(failure_reasons)

    return {
        "case_id": pred_case.get("case_id") or gold_case.get("case_id"),
        "case_category": case_category,
        "simplified_case_category": simplified_case_category,
        "failed": failed,
        "failure_reasons": failure_reasons,
        "recommendation": rec_scores,
        "avoid": avoid_scores,
        "caution": caution_scores,
        "alternative": alt_scores,
        "unsafe_recommendation": bool(unsafe_items),
        "unsafe_items": unsafe_items,
        "trace_pass": trace_pass,
        "unsupported_claim_rate": unsupported_claim_rate,
        "gold": gold,
        "prediction": pred,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _micro_scores(details: List[Dict[str, Any]], field: str) -> Dict[str, float]:
    tp = sum(item[field]["hit_count"] for item in details)
    pred_total = sum(item[field]["pred_count"] for item in details)
    gold_total = sum(item[field]["gold_count"] for item in details)
    precision = safe_divide(tp, pred_total)
    recall = safe_divide(tp, gold_total)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "hit_count": tp,
        "pred_count": pred_total,
        "gold_count": gold_total,
        "hit_fraction": f"{tp}/{gold_total}" if gold_total else "NA",
    }



def overall_safety_reasoning_score(summary: Dict[str, Any]) -> Dict[str, float]:
    """Compute an overall safety reasoning score (OSRS).

    OSRS is a weighted composite for publication-level summarization.
    It complements, but does not replace, the individual safety metrics.

    Formula:
        0.25 * M_avoid_recall
      + 0.20 * M_rec_recall
      + 0.15 * M_caution_recall
      + 0.15 * M_alt_recall
      + 0.15 * trace_pass_rate
      + 0.10 * (1 - unsafe_recommendation_rate)
    """
    micro = summary.get("micro", {})
    m_rec = float(micro.get("M_rec", {}).get("recall", 0.0))
    m_avoid = float(micro.get("M_avoid", {}).get("recall", 0.0))
    m_caution = float(micro.get("M_caution", {}).get("recall", 0.0))
    m_alt = float(micro.get("M_alt", {}).get("recall", 0.0))
    trace_pass = float(summary.get("trace_consistency_pass_rate", 0.0))
    unsafe_rate = float(summary.get("unsafe_recommendation_rate", 0.0))
    unsafe_penalty_component = max(0.0, 1.0 - unsafe_rate)

    components = {
        "M_avoid_recall": m_avoid,
        "M_rec_recall": m_rec,
        "M_caution_recall": m_caution,
        "M_alt_recall": m_alt,
        "trace_consistency_pass_rate": trace_pass,
        "unsafe_free_rate": unsafe_penalty_component,
    }
    weights = {
        "M_avoid_recall": 0.25,
        "M_rec_recall": 0.20,
        "M_caution_recall": 0.15,
        "M_alt_recall": 0.15,
        "trace_consistency_pass_rate": 0.15,
        "unsafe_free_rate": 0.10,
    }
    score = sum(weights[name] * components[name] for name in weights)
    return {
        "overall_safety_reasoning_score": score,
        "overall_safety_reasoning_score_percent": score * 100.0,
        "osrs_formula": (
            "0.25*M_avoid_recall + 0.20*M_rec_recall + "
            "0.15*M_caution_recall + 0.15*M_alt_recall + "
            "0.15*trace_pass_rate + 0.10*(1-unsafe_rate)"
        ),
        "osrs_weights": weights,
        "osrs_components": components,
    }


def evaluate(predictions: List[Dict[str, Any]], gold_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate predictions against gold rows.

    Returns a dict containing:
    - summary: aggregate metrics
    - details: per-case detailed scores
    """
    gold_by_id = {row["case_id"]: row for row in gold_rows}
    details: List[Dict[str, Any]] = []

    missing_gold_ids = []
    for pred in predictions:
        case_id = pred.get("case_id")
        gold = gold_by_id.get(case_id)
        if gold is None:
            missing_gold_ids.append(case_id)
            continue
        details.append(evaluate_case_detailed(pred, gold))

    total = len(details)
    failed_cases = [item for item in details if item["failed"]]
    success_cases = [item for item in details if not item["failed"]]

    category_counts = Counter(item["case_category"] for item in details)
    simplified_category_counts = Counter(item["simplified_case_category"] for item in details)
    failure_reason_counts = Counter(
        reason for item in details for reason in item["failure_reasons"]
    )

    # Backward-compatible case-level hit metrics.
    rec_case_hit = _mean([1.0 if item["recommendation"]["hit_count"] > 0 or item["recommendation"]["gold_count"] == 0 else 0.0 for item in details])
    avoid_case_hit = _mean([1.0 if item["avoid"]["hit_count"] > 0 or item["avoid"]["gold_count"] == 0 else 0.0 for item in details])
    caution_case_hit = _mean([1.0 if item["caution"]["hit_count"] > 0 or item["caution"]["gold_count"] == 0 else 0.0 for item in details])
    alt_case_hit = _mean([1.0 if item["alternative"]["hit_count"] > 0 or item["alternative"]["gold_count"] == 0 else 0.0 for item in details])

    summary = {
        "total_cases": total,
        "missing_gold_ids": missing_gold_ids,
        "success_cases": len(success_cases),
        "failed_cases": len(failed_cases),
        "success_rate_strict": _mean([0.0 if item["failed"] else 1.0 for item in details]),
        "failure_rate_strict": _mean([1.0 if item["failed"] else 0.0 for item in details]),
        "case_categories": dict(category_counts),
        "simplified_case_categories": dict(simplified_category_counts),
        "simplified_failure_taxonomy": {
            "success": "No safety-critical, evidence, or clinical-completeness failure under simplified taxonomy.",
            "safety_critical_failure": "Unsafe recommendation or missed avoid-level medication.",
            "evidence_failure": "Trace/evidence verification failed after safety-critical failures are excluded.",
            "clinical_completeness_failure": "Recommended, caution, or alternative output is incomplete after safety/evidence failures are excluded."
        },
        "failure_reason_counts": dict(failure_reason_counts),

        # Old case-level hit metrics for compatibility.
        "case_level_recommendation_hit_rate": rec_case_hit,
        "case_level_trap_avoid_hit_rate": avoid_case_hit,
        "case_level_caution_hit_rate": caution_case_hit,
        "safer_alternative_hit_rate": alt_case_hit,

        # New detailed macro metrics.
        "macro": {
            "M_rec_precision": _mean([item["recommendation"]["precision"] for item in details]),
            "M_rec_recall": _mean([item["recommendation"]["recall"] for item in details]),
            "M_rec_f1": _mean([item["recommendation"]["f1"] for item in details]),
            "M_avoid_precision": _mean([item["avoid"]["precision"] for item in details]),
            "M_avoid_recall": _mean([item["avoid"]["recall"] for item in details]),
            "M_avoid_f1": _mean([item["avoid"]["f1"] for item in details]),
            "M_caution_precision": _mean([item["caution"]["precision"] for item in details]),
            "M_caution_recall": _mean([item["caution"]["recall"] for item in details]),
            "M_caution_f1": _mean([item["caution"]["f1"] for item in details]),
            "M_alt_precision": _mean([item["alternative"]["precision"] for item in details]),
            "M_alt_recall": _mean([item["alternative"]["recall"] for item in details]),
            "M_alt_f1": _mean([item["alternative"]["f1"] for item in details]),
        },

        # New detailed micro metrics.
        "micro": {
            "M_rec": _micro_scores(details, "recommendation"),
            "M_avoid": _micro_scores(details, "avoid"),
            "M_caution": _micro_scores(details, "caution"),
            "M_alt": _micro_scores(details, "alternative"),
        },

        "unsafe_recommendation_count": sum(1 for item in details if item["unsafe_recommendation"]),
        "unsafe_recommendation_rate": _mean([1.0 if item["unsafe_recommendation"] else 0.0 for item in details]),
        "trace_consistency_pass_rate": _mean([1.0 if item["trace_pass"] else 0.0 for item in details]),
        "avg_unsupported_claim_rate": _mean([item["unsupported_claim_rate"] for item in details]),
        "failed_case_ids_strict": [item["case_id"] for item in failed_cases],
        "success_case_ids_strict": [item["case_id"] for item in success_cases],
    }

    # Ensure all simplified categories appear in summaries, even if count is zero.
    for category in [
        "success",
        "safety_critical_failure",
        "clinical_completeness_failure",
        "evidence_failure",
    ]:
        summary["simplified_case_categories"].setdefault(category, 0)

    # Publication-level composite score. This is a summary metric only;
    # individual safety metrics remain the primary evidence.
    summary["overall_safety_reasoning"] = overall_safety_reasoning_score(summary)

    return {"summary": summary, "details": details}
