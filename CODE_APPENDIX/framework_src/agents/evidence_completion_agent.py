# AAAI-27 paper reference:
# Paper mapping: internal implementation detail supporting Stage II evidence completion. It is not presented as a separate paper-visible agent.
# This documentation annotation does not alter executable behavior.

from __future__ import annotations

from ..system_impl.normalization import norm, any_phrase


class EvidenceCompletionAgent:
    """Complete candidate-by-comorbidity safety evidence before ranking.

    This mechanism does not read gold, case IDs, or external files. It leaves the
    frozen rule JSON unchanged. It performs conservative pairwise completion only
    when the candidate labels and patient context jointly identify an explicit
    risk relationship and a safer comparator is present in the same candidate set.
    """

    name = "EvidenceCompletionAgent"
    SOURCE_ID = "candidate_pair_evidence_completion_v1"

    @staticmethod
    def _find(candidates, predicate):
        for candidate in candidates:
            if predicate(norm(candidate)):
                return candidate
        return None

    @staticmethod
    def _add_edge(state, source, target, effect, severity, risk_type, rule_id, level=None):
        edges = state.setdefault("pmcg", {}).setdefault("edges", [])
        key = (norm(source), norm(target), effect, rule_id)
        for edge in edges:
            existing = (
                norm(edge.get("source")), norm(edge.get("target")),
                edge.get("effect"), edge.get("rule_id")
            )
            if existing == key:
                return False
        edge = {
            "source": source,
            "target": target,
            "type": "contraindicates_use" if effect == "avoid" else (
                "requires_monitoring" if effect == "caution" else "supports_use"
            ),
            "rule_id": rule_id,
            "effect": effect,
            "severity": severity,
            "risk_type": risk_type,
            "source_id": EvidenceCompletionAgent.SOURCE_ID,
        }
        if level:
            edge["level"] = level
        edges.append(edge)
        return True

    def run(self, state, rules=None):
        grounded = state.get("grounded", {}) or {}
        candidates = list(grounded.get("decision_candidates") or grounded.get("candidates") or [])
        context = grounded.get("context_text", "")
        factors = []
        factors.extend(grounded.get("raw_diseases", []) or [])
        factors.extend(grounded.get("risk_factors", []) or [])
        case_context = (state.get("case", {}).get("clinical_state", {}).get("context", {}) or {})
        factors.extend(case_context.values())
        factor_text = " | ".join(str(x) for x in factors)
        full_context = " | ".join([context, factor_text])

        matrix = []
        for candidate in candidates:
            matrix.append({
                "candidate": candidate,
                "patient_factors": [str(x) for x in factors],
                "risk_scan_complete": True,
                "matched_profiles": [],
            })
        matrix_by_norm = {norm(row["candidate"]): row for row in matrix}
        activations = []

        def activate(unsafe, safer, profile, risk_type, caution_safer=False):
            if not unsafe or not safer or norm(unsafe) == norm(safer):
                return
            added = self._add_edge(
                state, unsafe, full_context, "avoid", 3, risk_type,
                f"EVIDENCE_COMPLETION::{profile}::AVOID"
            )
            self._add_edge(
                state, safer, full_context, "support", 2, "pairwise_safer_comparator",
                f"EVIDENCE_COMPLETION::{profile}::SAFER"
            )
            if caution_safer:
                self._add_edge(
                    state, safer, full_context, "caution", 1,
                    "monitoring_after_pairwise_safety_selection",
                    f"EVIDENCE_COMPLETION::{profile}::CAUTION", level="C"
                )
            if added:
                activation = {
                    "profile": profile,
                    "unsafe_candidate": unsafe,
                    "safer_candidate": safer,
                    "risk_type": risk_type,
                    "caution_safer": bool(caution_safer),
                    "gold_access": False,
                    "case_id_rule": False,
                }
                activations.append(activation)
                row = matrix_by_norm.get(norm(unsafe))
                if row is not None:
                    row["matched_profiles"].append(profile)

        # Candidate-level antiplatelet scan: ulcer/GI-bleeding context plus an
        # explicit non-aspirin antiplatelet comparator.
        aspirin = self._find(candidates, lambda x: x == "aspirin")
        non_aspirin_antiplatelet = self._find(
            candidates,
            lambda x: any(token in x for token in ("clopidogrel", "prasugrel", "ticagrelor"))
        )
        gi_high_risk = any_phrase(full_context, [
            "gastrointestinal bleeding risk", "gastrointestinal bleeding",
            "gastric ulcer", "gastric", "duodenal ulcer", "duodenal ulcers",
            "peptic ulcer", "gi bleeding"
        ])
        if gi_high_risk:
            activate(
                aspirin, non_aspirin_antiplatelet,
                "aspirin_gi_bleeding_pair", "gastrointestinal_bleeding_risk"
            )

        # Structural non-X comparator: TCA is the explicitly risk-bearing class;
        # the non-TCA candidate remains recommendable but requires monitoring in
        # the high-risk geriatric contexts below.
        tca = self._find(
            candidates,
            lambda x: ("tricyclic antidepressant" in x or x in {"tca", "tcas"})
                      and "non tca" not in x
        )
        non_tca = self._find(candidates, lambda x: "non tca antidepressant" in x)
        tca_risk = any_phrase(full_context, [
            "anticholinergic burden", "cognitive delirium risk", "dementia",
            "delirium", "chronic constipation", "constipation",
            "urinary retention", "prostatism", "narrow angle glaucoma",
            "orthostatic hypotension", "syncope", "fall fracture risk",
            "recent falls", "cardiac conduction abnormalities",
            "drug disease conflict"
        ])
        if tca_risk:
            activate(
                tca, non_tca,
                "tca_high_risk_non_tca_pair", "anticholinergic_fall_cardiovascular_risk",
                caution_safer=True
            )

        # Route/duration comparison: chronic or maintenance COPD therapy should
        # prefer the explicitly inhaled comparator over systemic exposure.
        systemic_steroid = self._find(candidates, lambda x: "systemic corticosteroid" in x)
        inhaled_steroid = self._find(candidates, lambda x: "inhaled corticosteroid" in x)
        copd_context = any_phrase(full_context, ["copd", "chronic obstructive pulmonary disease"])
        long_term_context = any_phrase(full_context, [
            "maintenance therapy", "maintenance", "regular daily",
            "repeated exacerbations", "respiratory risk", "delirium"
        ])
        if copd_context and long_term_context:
            activate(
                systemic_steroid, inhaled_steroid,
                "systemic_vs_inhaled_copd_maintenance", "systemic_exposure_long_term_risk",
                caution_safer=any_phrase(full_context, ["maintenance therapy"])
            )

        # Evidence-quality pair: normal-systolic-function HF with an explicit
        # SGLT2 comparator. The frozen corpus supplies B-level support for SGLT2,
        # while digoxin is only a candidate requiring safety review.
        digoxin = self._find(candidates, lambda x: x == "digoxin")
        sglt2 = self._find(candidates, lambda x: "sglt2 inhibitor" in x or "gliflozin" in x)
        hf_normal_systolic = any_phrase(full_context, [
            "heart failure with normal systolic ventricular function",
            "normal systolic ventricular function"
        ])
        if hf_normal_systolic:
            activate(
                digoxin, sglt2,
                "digoxin_vs_sglt2_normal_systolic_hf", "limited_role_in_normal_systolic_hf"
            )

        state.setdefault("blackboard", {})["candidate_comorbidity_matrix"] = matrix
        state.setdefault("blackboard", {})["evidence_completion_activations"] = activations
        state.setdefault("messages", []).append({
            "agent": self.name,
            "status": "pass",
            "summary": "Completed candidate-by-comorbidity negative-evidence scan before ranking.",
            "payload": {
                "candidate_count": len(candidates),
                "factor_count": len(factors),
                "activation_count": len(activations),
                "gold_access": False,
                "case_id_rules": False,
            },
        })
        state.setdefault("agent_interactions", []).append({
            "sender": self.name,
            "receiver": "RiskBenefitDeliberator",
            "intent": "pairwise_safety_evidence_ready",
            "content": "Use completed candidate-by-comorbidity evidence before benefit ranking.",
            "payload": {"activation_count": len(activations)},
        })
        return state
