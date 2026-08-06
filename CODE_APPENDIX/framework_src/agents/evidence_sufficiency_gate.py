# AAAI-27 paper reference:
# Paper mapping: Stage IV final decision verification. This confidence-to-coverage guard is an internal part of the paper-visible Safety Gate; see PAPER_COMPONENT_MAPPING.md.
# This documentation annotation does not alter executable behavior.

from __future__ import annotations


class EvidenceSufficiencyGate:
    """Withhold disproportionate-confidence recommendations pending review.

    The threshold was frozen from the 95th percentile of the Dev39
    confidence-to-coverage ratio. The gate uses only inference-time state. It
    does not read gold, case IDs, datasets, or external files, and it never
    overrides an avoid-level safety decision.

    Ratio:
        minimum recommendation margin / total PMCG edge count

    A very high ratio indicates that a strong definitive recommendation is
    being made from a comparatively narrow patient-specific graph. The gate
    conservatively withholds the definitive M_rec output while preserving the
    existing avoid, caution, alternative, and trace structures.
    """

    name = "EvidenceSufficiencyGate"
    SOURCE_ID = "dev39_p95_confidence_coverage_guard_v1"
    # Frozen from Dev39 only (linear 95th percentile): 0.1986607142857143.
    THRESHOLD = 0.1986607142857143

    def run(self, state, rules=None):
        decision = state.get("decision") or {}
        recs = list(decision.get("M_rec", []) or [])
        rationale = decision.get("yes_no_rationale", {}) or {}
        edge_count = len((state.get("pmcg") or {}).get("edges", []) or [])

        margins = []
        for candidate in recs:
            row = rationale.get(candidate, {}) or {}
            support = float(row.get("score_support", 0.0) or 0.0)
            harm = float(row.get("score_harm", 0.0) or 0.0)
            margins.append(support - harm)

        min_margin = min(margins) if margins else 0.0
        ratio = min_margin / max(edge_count, 1)
        triggered = bool(recs) and ratio > self.THRESHOLD
        withheld = []

        if triggered:
            withheld = list(recs)
            decision["M_rec"] = []
            decision["U"] = True
            decision["uncertainty_reason"] = (
                "definitive recommendation withheld because confidence was "
                "disproportionate to patient-specific PMCG coverage"
            )
            state["decision"] = decision

        audit = {
            "threshold": self.THRESHOLD,
            "ratio": ratio,
            "minimum_recommendation_margin": min_margin,
            "pmcg_edge_count": edge_count,
            "triggered": triggered,
            "withheld_candidates": withheld,
            "gold_access": False,
            "case_id_rule": False,
            "hard_safety_override": False,
            "source_id": self.SOURCE_ID,
        }
        state.setdefault("blackboard", {})["evidence_sufficiency_gate"] = audit
        state.setdefault("messages", []).append({
            "agent": self.name,
            "status": "review" if triggered else "pass",
            "summary": (
                "Withheld a disproportionate-confidence recommendation pending review."
                if triggered else
                "Recommendation confidence was proportionate to PMCG coverage."
            ),
            "payload": audit,
        })
        state.setdefault("agent_interactions", []).append({
            "sender": self.name,
            "receiver": "TraceVerifier",
            "intent": "evidence_sufficiency_checked",
            "content": "Verify the final decision after the confidence-to-coverage guard.",
            "payload": {"triggered": triggered, "withheld_count": len(withheld)},
        })
        return state
