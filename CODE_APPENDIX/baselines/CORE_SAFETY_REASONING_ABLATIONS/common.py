from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]

FINAL_INPUT = ROOT / "02_FROZEN_INFERENCE" / "data" / "final_test_input.json"
FROZEN_RULES = ROOT / "02_FROZEN_INFERENCE" / "frozen" / "frozen_rules.json"
ATLAS_PRED = ROOT / "02_FROZEN_INFERENCE" / "outputs" / "final_predictions.json"

OUT_DIR = ROOT / "baselines_v2" / "outputs"
RESULT_DIR = ROOT / "baselines_v2" / "results"
LOG_DIR = ROOT / "baselines_v2" / "logs"

for p in [OUT_DIR, RESULT_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_text(x: Any) -> str:
    s = str(x or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def loose_match(a: Any, b: Any) -> bool:
    aa = norm_text(a)
    bb = norm_text(b)
    if not aa or not bb:
        return False
    return aa in bb or bb in aa


def candidate_names(case: Dict[str, Any]) -> List[str]:
    out = []
    for item in case.get("candidate_medications", []) or []:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]).strip())
        elif isinstance(item, str) and item.strip():
            out.append(item.strip())

    final = []
    seen = set()
    for x in out:
        k = norm_text(x)
        if k and k not in seen:
            seen.add(k)
            final.append(x)
    return final


def flatten_case_text(case: Dict[str, Any]) -> str:
    vals = []

    patient = case.get("patient", {}) or {}
    vals.extend([str(v) for v in patient.values() if v is not None])

    cs = case.get("clinical_state", {}) or {}
    for k in ["diseases", "current_medications", "age_related_factors", "risk_factors"]:
        vals.extend([str(x) for x in cs.get(k, []) or []])

    ctx = cs.get("context", {}) or {}
    if isinstance(ctx, dict):
        vals.extend([str(v) for v in ctx.values() if v is not None])

    vals.extend(candidate_names(case))
    vals.append(str(case.get("task", "")))
    vals.append("older adult")

    return " | ".join(vals)


def unique_keep_order(xs: List[Any]) -> List[str]:
    out = []
    seen = set()
    for x in xs or []:
        s = str(x or "").strip()
        if not s:
            continue
        k = norm_text(s)
        if k and k not in seen:
            seen.add(k)
            out.append(s.lower())
    return out


def make_prediction(
    case_id: str,
    M_rec=None,
    M_avoid=None,
    M_caution=None,
    M_alt=None,
    M_level=None,
    reasoning=None,
    sources=None,
    method: str = "",
    trace_pass: bool = True,
) -> Dict[str, Any]:
    M_rec = unique_keep_order(M_rec or [])
    M_avoid = unique_keep_order(M_avoid or [])
    M_caution = unique_keep_order(M_caution or [])
    M_alt = unique_keep_order(M_alt or [])

    avoid_keys = {norm_text(x) for x in M_avoid}
    rec_keys = {norm_text(x) for x in M_rec}

    # Safety cleanup: do not recommend predicted avoid drugs.
    M_rec = [x for x in M_rec if norm_text(x) not in avoid_keys]
    M_caution = [
        x for x in M_caution
        if norm_text(x) not in avoid_keys and norm_text(x) not in rec_keys
    ]

    if not isinstance(M_level, dict):
        M_level = {}

    clean_level = {}
    for k, v in M_level.items():
        kk = str(k or "").strip().lower()
        if kk:
            clean_level[kk] = str(v or "B")

    for m in M_rec:
        clean_level.setdefault(m, "B")

    return {
        "case_id": case_id,
        "final_decision": {
            "M_rec": M_rec,
            "M_avoid": M_avoid,
            "M_caution": M_caution,
            "M_alt": M_alt,
            "M_level": clean_level,
            "E": {
                "reasoning_path": [str(x) for x in (reasoning or [])][:20],
                "explanation": f"{method} baseline prediction.",
            },
            "S": [str(x) for x in (sources or [method])][:20],
            "U": False,
        },
        "trace_verification": {
            "trace_consistency": "pass" if trace_pass else "fail",
            "unsupported_claim_rate": 0.0 if trace_pass else 1.0,
        },
        "audit": {
            "baseline_schema": "baselines_v2_first_batch",
            "method": method,
            "uses_final_gold_during_inference": False,
        },
    }
