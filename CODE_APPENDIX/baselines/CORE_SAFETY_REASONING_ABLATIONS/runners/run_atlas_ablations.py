from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from framework_src.agents.unified_orchestrator import UnifiedOrchestratorAgent
from framework_src.system_impl.agents_impl import (
    TherapeuticNeedIdentifier,
    RiskBenefitDeliberator as BaseRiskBenefitDeliberator,
    SafetyFirstCandidatePruner,
    YesNoDecisionAgent,
    _append_message,
    _send,
)
from framework_src.system_impl.normalization import norm


FINAL_INPUT = ROOT / "02_FROZEN_INFERENCE" / "data" / "final_test_input.json"
FROZEN_RULES = ROOT / "02_FROZEN_INFERENCE" / "frozen" / "frozen_rules.json"
OUT_DIR = ROOT / "baselines_v2" / "outputs"


VARIANTS = [
    "no_pmcg",
    "no_drug_conflict_auditor",
    "no_geriatric_risk_auditor",
    "no_safety_gate",
    "no_alternative_agent",
    "no_safety_critic_revision",
    "no_safety_critic_only",
    "no_open_candidate_generation",
    "no_trace_verifier",
]


def _empty_pmcg():
    return {
        "nodes": [],
        "edges": [],
        "schema": {"node_types": [], "edge_types": []},
        "ablation_note": "PMCG disabled for ablation",
    }


class EmptyPMCGAgent:
    name = "PMCGBuilder_DISABLED"

    def run(self, state, rules=None):
        state["pmcg"] = _empty_pmcg()
        state.setdefault("blackboard", {})["pmcg_disabled"] = True
        _append_message(state, self.name, "disabled", "PMCG construction disabled for ablation.")
        return state

    def refresh_after_audits(self, state, rules=None):
        state["pmcg"] = state.get("pmcg") or _empty_pmcg()
        _append_message(state, self.name, "disabled", "PMCG audit refresh disabled for ablation.")
        return state


class NoDrugConflictAuditor:
    name = "DrugConflictAuditor_DISABLED"
    DROP_TYPES = {"contraindicates_use", "conflicts_with", "increases_risk_of"}

    def run(self, state, rules=None):
        if state.get("pmcg"):
            kept = []
            removed = []
            for e in state["pmcg"].get("edges", []):
                if e.get("effect") == "avoid" and e.get("type") in self.DROP_TYPES:
                    removed.append(e)
                else:
                    kept.append(e)
            state["pmcg"]["edges"] = kept
            state.setdefault("blackboard", {})["drug_conflict_edges_removed"] = len(removed)

        state["conflict_audit"] = {
            c: {
                "drug_disease_conflicts": [],
                "contraindication_count": 0,
                "conflict_count": 0,
            }
            for c in state.get("grounded", {}).get("candidates", [])
        }

        state.setdefault("blackboard", {})["conflict_audit"] = state["conflict_audit"]
        _append_message(
            state,
            self.name,
            "disabled",
            "Drug conflict auditor and avoid-conflict evidence channel disabled.",
        )
        return state


class NoGeriatricRiskAuditor:
    name = "GeriatricRiskAuditor_DISABLED"

    def run(self, state, rules=None):
        if state.get("pmcg"):
            kept = []
            removed = []
            for e in state["pmcg"].get("edges", []):
                risk_text = " ".join(
                    str(e.get(k, ""))
                    for k in ("risk_type", "rule_id", "source_id", "type")
                ).lower()

                is_geriatric = (
                    e.get("effect") == "caution"
                    or "geriatric" in risk_text
                    or "age" in risk_text
                    or "fall" in risk_text
                    or "renal" in risk_text
                    or "bleed" in risk_text
                    or "cognitive" in risk_text
                    or "anticholinergic" in risk_text
                    or "monitor" in risk_text
                )

                if is_geriatric:
                    removed.append(e)
                else:
                    kept.append(e)

            state["pmcg"]["edges"] = kept
            state.setdefault("blackboard", {})["geriatric_edges_removed"] = len(removed)

        state["geriatric_audit"] = {
            c: {
                "risk_edges": [],
                "risk_types": [],
                "age_amplified": False,
                "conflict_context_seen": False,
            }
            for c in state.get("grounded", {}).get("candidates", [])
        }

        state.setdefault("blackboard", {})["geriatric_audit"] = state["geriatric_audit"]
        _append_message(
            state,
            self.name,
            "disabled",
            "Geriatric risk auditor and caution/monitoring evidence channel disabled.",
        )
        return state


class CandidateGenerationNoOpen:
    name = "GuidelineCandidateAgent_NO_OPEN_GENERATION"

    def __init__(self):
        self.need = TherapeuticNeedIdentifier()

    def run(self, state, rules=None):
        state = self.need.run(state)
        g = state["grounded"]
        seeds = g.get("seed_candidates", [])[:]

        g["candidate_pool"] = seeds[:]
        g["candidates"] = seeds[:]
        g["decision_candidates"] = seeds[:]

        state["open_candidate_generation"] = {
            "policy": "disabled_for_ablation",
            "seed_candidates": seeds,
            "generated_candidate_pool": seeds,
            "generated_candidate_records": [],
            "free_text_generation": False,
            "decision_candidates_for_current_benchmark": seeds,
        }

        state.setdefault("blackboard", {})["open_candidate_generation"] = state["open_candidate_generation"]

        _append_message(
            state,
            self.name,
            "disabled",
            "Open candidate expansion disabled; using seed candidates only.",
            {"seed_count": len(seeds)},
        )
        return state


class NoAlternativeAgent:
    name = "AlternativeAgent_DISABLED"

    def run(self, state, rules=None):
        state["alternative_selection"] = {
            "choice_A_knowledge_constrained": False,
            "safer_alternative": None,
            "unsafe_reference": None,
            "decision_candidates": state.get("grounded", {}).get("decision_candidates", []),
            "ablation_note": "Alternative search disabled",
        }

        state.setdefault("blackboard", {})["alternative_selection"] = state["alternative_selection"]

        _append_message(
            state,
            self.name,
            "disabled",
            "Safer alternative search disabled for ablation.",
        )
        return state


class NoSafetyGateRiskBenefitDeliberator:
    name = "RiskBenefitDeliberator_NO_SAFETY_GATE"

    def run(self, state, rules=None):
        state.pop("safety_pruning", None)
        state.pop("yes_no_decision_board", None)
        state["yes_no_judgments"] = []

        candidates = (
            state.get("grounded", {}).get("decision_candidates")
            or state.get("grounded", {}).get("candidates", [])
        )
        edges = state.get("pmcg", {}).get("edges", [])

        scored = []

        for c in candidates:
            c_edges = [e for e in edges if e.get("source") == c]
            support = sum(
                float(e.get("severity", 1) or 1)
                for e in c_edges
                if e.get("effect") == "support"
            )
            harm = sum(
                float(e.get("severity", 1) or 1)
                for e in c_edges
                if e.get("effect") in ("avoid", "caution")
            )

            scored.append({
                "candidate": c,
                "support_only_score": support,
                "ignored_harm_score": harm,
                "edges": c_edges,
            })

        if not scored:
            state["decision"] = {
                "M_rec": [],
                "M_avoid": [],
                "M_caution": [],
                "M_alt": [],
                "M_level": {},
                "U": False,
                "uncertainty_reason": "safety gate disabled",
            }
            return state

        ordered = sorted(
            scored,
            key=lambda x: (x["support_only_score"], norm(x["candidate"])),
            reverse=True,
        )

        rec = ordered[0]
        avoid = ordered[-1] if len(ordered) > 1 else ordered[0]

        state["decision"] = {
            "M_rec": [rec["candidate"]],
            "M_avoid": [avoid["candidate"]] if norm(avoid["candidate"]) != norm(rec["candidate"]) else [],
            "M_caution": [],
            "M_alt": [rec["candidate"]],
            "M_level": {rec["candidate"]: "UNKNOWN"},
            "U": False,
            "uncertainty_reason": "safety gate disabled: benefit/support-only ranking",
            "yes_no_rationale": {},
        }

        state["scores"] = scored
        state.setdefault("blackboard", {})["safety_gate_disabled"] = True
        state.setdefault("blackboard", {})["decision_draft"] = state["decision"]

        _append_message(
            state,
            self.name,
            "disabled",
            "Safety gate disabled; made support-only decision without avoid/caution hard gating.",
            {"scores": [{k: v for k, v in x.items() if k != "edges"} for x in scored]},
        )

        _send(
            state,
            self.name,
            "TraceConsistencyVerifier",
            "decision_ready_for_verification",
            "Support-only draft decision is ready for trace verification.",
        )

        return state


class NoAlternativeRiskBenefitDeliberator:
    name = "RiskBenefitDeliberator_NO_ALTERNATIVE"

    def __init__(self):
        self.pruner = SafetyFirstCandidatePruner()
        self.yesno = YesNoDecisionAgent()
        self.impl = BaseRiskBenefitDeliberator()

    def run(self, state, rules=None):
        state = self.pruner.run(state)
        state = self.yesno.run(state)
        state = self.impl.run(state)

        if state.get("decision"):
            state["decision"]["M_alt"] = []

        state["alternative_selection"] = {
            "choice_A_knowledge_constrained": False,
            "safer_alternative": None,
            "unsafe_reference": None,
            "ablation_note": "Alternative search disabled after deliberation",
        }

        state.setdefault("blackboard", {})["alternative_selection"] = state["alternative_selection"]

        _append_message(
            state,
            self.name,
            "disabled",
            "Deliberation completed without safer-alternative module; M_alt cleared.",
        )
        return state


class TraceVerifierDisabled:
    name = "TraceVerifier_DISABLED"

    def run(self, state, rules=None):
        state["trace_verification"] = {
            "trace_consistency": "disabled",
            "recommendation_supported": None,
            "avoidance_supported": None,
            "disjoint_outputs": None,
            "alternative_consistent": None,
            "unsupported_claims": [],
            "unsupported_claim_rate": 0.0,
            "ablation_note": "Trace verification disabled; trace pass should not be credited.",
        }

        state.setdefault("blackboard", {})["trace_verification"] = state["trace_verification"]

        _append_message(
            state,
            self.name,
            "disabled",
            "Trace verification disabled for ablation.",
        )
        return state


class AblatedOrchestrator(UnifiedOrchestratorAgent):
    def __init__(self, rules, ablation: str, max_rounds: int = 4):
        super().__init__(rules, max_rounds=max_rounds)

        if ablation not in VARIANTS:
            raise ValueError(f"Unknown ablation: {ablation}. Choose from {VARIANTS}")

        self.ablation = ablation

        if ablation == "no_pmcg":
            self.registry["pmcg_building"] = EmptyPMCGAgent()

        elif ablation == "no_drug_conflict_auditor":
            self.registry["drug_conflict_audit"] = NoDrugConflictAuditor()

        elif ablation == "no_geriatric_risk_auditor":
            self.registry["geriatric_risk_audit"] = NoGeriatricRiskAuditor()

        elif ablation == "no_open_candidate_generation":
            self.registry["candidate_generation"] = CandidateGenerationNoOpen()

        elif ablation == "no_safety_gate":
            self.registry["risk_benefit_deliberation"] = NoSafetyGateRiskBenefitDeliberator()

        elif ablation == "no_alternative_agent":
            self.registry["alternative_search"] = NoAlternativeAgent()
            self.registry["risk_benefit_deliberation"] = NoAlternativeRiskBenefitDeliberator()

        elif ablation == "no_trace_verifier":
            self.registry["trace_verification"] = TraceVerifierDisabled()

    def _initial_tasks(self):
        tasks = super()._initial_tasks()

        if self.ablation == "no_safety_critic_revision":
            tasks = [
                t for t in tasks
                if t not in {"safety_critique", "decision_revision"}
            ]

        return tasks

    def _needs_review(self, state):
        tasks = super()._needs_review(state)

        if self.ablation == "no_safety_critic_revision":
            tasks = [
                t for t in tasks
                if t not in {"safety_critique", "decision_revision"}
            ]

        if self.ablation == "no_trace_verifier":
            tasks = [
                t for t in tasks
                if t != "trace_verification"
            ]

        return tasks

    def run_case(self, case):
        pred = super().run_case(case)
        pred["ablation"] = self.ablation
        pred.setdefault("ablation_metadata", {})["disabled_component"] = self.ablation
        return pred


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", required=True, choices=VARIANTS)
    parser.add_argument("--name", default=None)
    parser.add_argument("--max-rounds", type=int, default=4)
    args = parser.parse_args()

    cases = load_json(FINAL_INPUT)
    rules = load_json(FROZEN_RULES)

    name = args.name or f"ablation_{args.ablation}"

    preds = []
    orch = AblatedOrchestrator(
        rules,
        ablation=args.ablation,
        max_rounds=args.max_rounds,
    )

    for i, case in enumerate(cases, start=1):
        pred = orch.run_case(case)
        preds.append(pred)

        if i % 25 == 0 or i == len(cases):
            print(f"[{args.ablation}] {i}/{len(cases)}")

    out_path = OUT_DIR / f"final_predictions_{name}.json"
    fail_path = OUT_DIR / f"failures_{name}.json"

    save_json(preds, out_path)
    save_json([], fail_path)

    print(f"[OK] wrote {len(preds)} predictions: {out_path}")
    print(f"[OK] wrote failures: {fail_path}")


if __name__ == "__main__":
    main()
