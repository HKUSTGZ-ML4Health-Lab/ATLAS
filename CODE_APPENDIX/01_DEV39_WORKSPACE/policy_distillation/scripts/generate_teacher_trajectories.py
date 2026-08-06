#!/usr/bin/env python3
"""Generate deterministic Dev39 teacher trajectories for Stage-1 distillation.

This script reads only Dev39 inputs and the Dev39 working policy. It never reads
an evaluation reference. It records compact teacher traces under bounded review
budgets K in {1,2,3}. The reported experiments continue to use the unchanged
frozen policy in CODE_APPENDIX/02_FROZEN_INFERENCE/frozen/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT = Path(__file__).resolve()
PD_ROOT = SCRIPT.parents[1]
DEV_ROOT = PD_ROOT.parent
CODE_APPENDIX = DEV_ROOT.parent
sys.path.insert(0, str(CODE_APPENDIX))

from framework_src.agents.unified_orchestrator import UnifiedOrchestratorAgent  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def canonical_line(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unique_questions(judgments: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep unique internal teacher questions without changing their answers."""
    seen = set()
    output: List[Dict[str, Any]] = []
    for item in judgments:
        key = (
            str(item.get("agent", "")),
            str(item.get("question", "")),
            str(item.get("target", "")),
            bool(item.get("answer")),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                "scope": "teacher_internal_decision_check",
                "agent": item.get("agent"),
                "question": item.get("question"),
                "answer": bool(item.get("answer")),
                "answer_text": item.get("answer_text"),
                "target": item.get("target"),
                "evidence": item.get("evidence") or {},
                "rationale": item.get("rationale"),
            }
        )
    return output


def task_sequence(log: Iterable[Dict[str, Any]]) -> List[str]:
    return [
        str(entry.get("payload", {}).get("task"))
        for entry in log
        if entry.get("event") == "dispatch" and entry.get("payload", {}).get("task")
    ]


def pmcg_transitions(log: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pending: Dict[int, List[Dict[str, Any]]] = {}
    transitions: List[Dict[str, Any]] = []
    for entry in log:
        round_id = int(entry.get("round", 0) or 0)
        payload = entry.get("payload") or {}
        if entry.get("event") == "dispatch" and payload.get("task") == "pmcg_building":
            pending.setdefault(round_id, []).append(payload.get("before") or {})
            continue
        if entry.get("event") not in {"state_changed", "state_unchanged"}:
            continue
        if payload.get("task") != "pmcg_building":
            continue
        before_list = pending.get(round_id) or [{}]
        before = before_list.pop(0)
        after = payload.get("after") or {}
        transitions.append(
            {
                "round": round_id,
                "event": entry.get("event"),
                "edge_count_before": int(before.get("pmcg_edges", 0) or 0),
                "edge_count_after": int(after.get("pmcg_edges", 0) or 0),
                "candidate_count": int(after.get("candidate_pool", 0) or 0),
            }
        )
    return transitions


def selected_messages(messages: Iterable[Dict[str, Any]], agents: set[str]) -> List[Dict[str, Any]]:
    output = []
    for message in messages:
        if str(message.get("agent")) not in agents:
            continue
        output.append(
            {
                "agent": message.get("agent"),
                "status": message.get("status"),
                "summary": message.get("summary"),
                "payload": message.get("payload") or {},
            }
        )
    return output


def stopping_decision(log: Iterable[Dict[str, Any]], budget: int) -> Dict[str, Any]:
    converged = [entry for entry in log if entry.get("event") == "converged"]
    review_rounds = sorted({int(entry.get("round", 0) or 0) for entry in log if int(entry.get("round", 0) or 0) > 0})
    if converged:
        return {
            "status": "converged",
            "round": int(converged[-1].get("round", 0) or 0),
            "teacher_review_budget": budget,
        }
    return {
        "status": "review_budget_exhausted",
        "round": max(review_rounds, default=0),
        "teacher_review_budget": budget,
    }


def core_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["M_rec", "M_avoid", "M_caution", "M_alt", "M_level", "U", "uncertainty_reason"]
    return {key: decision.get(key) for key in keys}


def input_snapshot(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patient": case.get("patient") or {},
        "clinical_state": case.get("clinical_state") or {},
        "candidate_medications": case.get("candidate_medications") or [],
        "task": case.get("task"),
        "input_safety": case.get("input_safety") or {},
    }


def build_trajectory(case: Dict[str, Any], rules: Dict[str, Any], budget: int, provenance: Dict[str, str]) -> Dict[str, Any]:
    result = UnifiedOrchestratorAgent(rules, max_rounds=budget).run_case(case)
    messages = result.get("agent_messages") or []
    log = result.get("orchestration_log") or []
    edges = (result.get("pmcg") or {}).get("edges") or []
    activated_rule_ids = sorted({str(edge.get("rule_id")) for edge in edges if edge.get("rule_id")})
    effects: Dict[str, int] = {}
    for edge in edges:
        effect = str(edge.get("effect") or "unknown")
        effects[effect] = effects.get(effect, 0) + 1

    return {
        "trajectory_id": f"{case['case_id']}_k{budget}",
        "development_case_id": case["case_id"],
        "teacher_review_budget": budget,
        "teacher_mode": "bounded_multi_agent_review",
        "input_snapshot": input_snapshot(case),
        "questions": unique_questions(result.get("yes_no_judgments") or []),
        "patient_state_updates": selected_messages(messages, {"ClinicalStateGrounder"}),
        "pmcg_transitions": pmcg_transitions(log),
        "risk_assessments": selected_messages(
            messages,
            {
                "MedicationConflictAuditor",
                "GeriatricRiskAuditor",
                "SafetyFirstCandidatePruner",
                "YesNoDecisionAgent",
                "RiskBenefitDeliberator",
                "SafetyCritic",
                "EvidenceSufficiencyGate",
            },
        ),
        "decision_revisions": selected_messages(messages, {"RevisionAgent"}),
        "stopping_decision": stopping_decision(log, budget),
        "evidence_paths": {
            "source_ids": result.get("evidence_sources") or [],
            "risk_types": result.get("risk_types") or [],
            "pmcg_edge_effect_counts": dict(sorted(effects.items())),
            "trace_verification": result.get("trace_verification") or {},
        },
        "activated_rule_ids": activated_rule_ids,
        "task_sequence": task_sequence(log),
        "final_decision": core_decision(result.get("final_decision") or {}),
        "provenance": provenance,
    }


def main() -> None:
    config_path = PD_ROOT / "teacher_config.json"
    cases_path = DEV_ROOT / "data" / "dev_input.json"
    rules_path = DEV_ROOT / "rules" / "working_rules.json"
    output_path = PD_ROOT / "trajectories" / "dev39_k1_k2_k3.jsonl"

    config = load_json(config_path)
    cases = load_json(cases_path)
    rules = load_json(rules_path)
    budgets = [int(value) for value in config["teacher_review_budgets"]]

    if len(cases) != int(config["development_case_count"]):
        raise ValueError(f"Expected {config['development_case_count']} development cases, found {len(cases)}")
    if len(cases) * len(budgets) != int(config["expected_trajectory_count"]):
        raise ValueError("The configured case and budget counts do not produce the expected trajectory count")

    provenance = {
        "development_input_sha256": sha256(cases_path),
        "teacher_policy_sha256": sha256(rules_path),
        "teacher_config_sha256": sha256(config_path),
        "evaluation_reference_access": "false",
        "reported_inference_policy_modified": "false",
    }

    records: List[Dict[str, Any]] = []
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        for budget in budgets:
            records.append(build_trajectory(case, rules, budget, provenance))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(canonical_line(record) for record in records) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {len(records)} deterministic teacher trajectories to {output_path}")


if __name__ == "__main__":
    main()
