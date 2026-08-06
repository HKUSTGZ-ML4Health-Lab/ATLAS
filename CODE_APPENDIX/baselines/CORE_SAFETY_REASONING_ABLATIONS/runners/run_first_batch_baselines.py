from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from baselines_v2.common import (
    FINAL_INPUT,
    FROZEN_RULES,
    OUT_DIR,
    candidate_names,
    flatten_case_text,
    load_json,
    loose_match,
    make_prediction,
    norm_text,
    save_json,
    unique_keep_order,
)


def rule_context_match(rule: Dict[str, Any], case_text: str) -> bool:
    ctxs = rule.get("context_patterns", []) or []
    if not ctxs:
        return True
    return any(loose_match(case_text, x) for x in ctxs)


def rule_candidate_match(rule: Dict[str, Any], cand: str) -> bool:
    pats = rule.get("candidate_patterns", []) or []
    return any(loose_match(cand, p) for p in pats)


def alt_for_avoids(
    rules: Dict[str, Any],
    avoids: List[str],
    allow_candidate_rec: List[str] | None = None,
) -> List[str]:
    raw = rules.get("target_raw_rules", {}) or {}
    out = []

    for avoid in avoids:
        for r in raw.get("alternative_rules", []) or []:
            unsafe = r.get("unsafe_medications", []) or r.get("if_avoid_drug_contains", []) or []
            if not any(loose_match(avoid, u) for u in unsafe):
                continue

            alts = []
            if r.get("alternative"):
                alts.append(r["alternative"])
            alts.extend(r.get("alternatives", []) or [])
            out.extend(alts)

    # Fixed schema projection: if no explicit textual alternative is retrieved,
    # the selected non-avoided recommendation can serve as the safer alternative.
    if allow_candidate_rec and not out:
        out.extend(allow_candidate_rec)

    return unique_keep_order(out)


def frozen_guideline(cases: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    krules = rules.get("knowledge_rules", []) or []
    weights = {"support": 4.0, "avoid": 8.0, "caution": 2.0}
    preds = []

    for case in cases:
        cands = candidate_names(case)
        case_text = flatten_case_text(case)

        scores = {
            c: {
                "support": 0.0,
                "avoid": 0.0,
                "caution": 0.0,
                "sources": [],
                "level": "B",
            }
            for c in cands
        }

        reasoning = []

        for r in krules:
            if not rule_context_match(r, case_text):
                continue

            effect = str(r.get("effect", "")).lower()
            if effect not in {"support", "avoid", "caution"}:
                continue

            for c in cands:
                if not rule_candidate_match(r, c):
                    continue

                scores[c][effect] += weights[effect] * float(r.get("severity", 1) or 1)

                if r.get("source"):
                    scores[c]["sources"].append(r["source"])

                if r.get("level"):
                    scores[c]["level"] = r["level"]

                reasoning.append(f"{effect} rule matched for {c}: {r.get('id', 'rule')}")

        M_avoid = [c for c in cands if scores[c]["avoid"] > 0]
        avoid_keys = {norm_text(x) for x in M_avoid}

        M_caution = [
            c for c in cands
            if scores[c]["caution"] > 0 and norm_text(c) not in avoid_keys
        ]

        rec_pool = [
            c for c in cands
            if scores[c]["support"] > 0 and norm_text(c) not in avoid_keys
        ]
        rec_pool.sort(
            key=lambda c: scores[c]["support"] - 0.5 * scores[c]["caution"],
            reverse=True,
        )

        M_rec = rec_pool[:1]
        M_alt = alt_for_avoids(rules, M_avoid, allow_candidate_rec=M_rec)

        sources = []
        for c in cands:
            sources.extend(scores[c]["sources"])

        preds.append(
            make_prediction(
                case_id=case["case_id"],
                M_rec=M_rec,
                M_avoid=M_avoid,
                M_caution=M_caution,
                M_alt=M_alt,
                M_level={m: scores[m]["level"] for m in M_rec},
                reasoning=reasoning or ["No frozen guideline rule matched; abstained."],
                sources=unique_keep_order(sources) or ["frozen_rules"],
                method="Frozen Guideline Engine",
                trace_pass=True,
            )
        )

    return preds


def build_evidence_docs(rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = []
    raw = rules.get("target_raw_rules", {}) or {}

    for idx, rule in enumerate(raw.get("guideline_rules", []) or []):
        diseases = rule.get("if_disease_contains", []) or []
        recs = rule.get("recommend", []) or []

        for rec in recs:
            docs.append(
                {
                    "doc_id": f"guideline_{idx}_{norm_text(rec)[:40]}",
                    "action": "rec",
                    "target": rec,
                    "source": rule.get("source", rule.get("rule_id", "guideline_rule")),
                    "level": (rule.get("level", {}) or {}).get(rec, "B"),
                    "text": " ".join(
                        [
                            "recommend",
                            " ".join(map(str, diseases)),
                            str(rec),
                            str(rule.get("explanation", "")),
                        ]
                    ),
                }
            )

    for idx, rule in enumerate(raw.get("safety_rules", []) or []):
        drugs = rule.get("if_drug_contains", []) or rule.get("medications", []) or []
        triggers = (
            rule.get("if_risk_factor_contains", [])
            or rule.get("if_disease_contains", [])
            or rule.get("if_age_group", [])
            or ["older adult"]
        )

        bias = str(rule.get("decision_bias", rule.get("decision", ""))).lower()
        severity = str(rule.get("severity", "")).lower()
        action = "avoid" if ("avoid" in bias or "high" in severity) else "caution"

        for d in drugs:
            docs.append(
                {
                    "doc_id": f"safety_{action}_{idx}_{norm_text(d)[:40]}",
                    "action": action,
                    "target": d,
                    "source": rule.get("source", rule.get("rule_id", "safety_rule")),
                    "level": None,
                    "text": " ".join(
                        [
                            action,
                            " ".join(map(str, triggers)),
                            str(d),
                            bias,
                            severity,
                            str(rule.get("explanation", "")),
                        ]
                    ),
                }
            )

    for idx, rule in enumerate(raw.get("alternative_rules", []) or []):
        unsafe = rule.get("unsafe_medications", []) or rule.get("if_avoid_drug_contains", []) or []

        alts = []
        if rule.get("alternative"):
            alts.append(rule["alternative"])
        alts.extend(rule.get("alternatives", []) or [])

        for u in unsafe:
            for alt in alts:
                docs.append(
                    {
                        "doc_id": f"alt_{idx}_{norm_text(u)[:25]}_{norm_text(alt)[:25]}",
                        "action": "alt",
                        "unsafe": u,
                        "target": alt,
                        "source": rule.get("source", rule.get("rule_id", "alternative_rule")),
                        "level": None,
                        "text": " ".join(["alternative", str(u), str(alt), str(rule.get("explanation", ""))]),
                    }
                )

    return docs


def retrieval_only(cases: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = build_evidence_docs(rules)
    if not docs:
        raise RuntimeError("No evidence docs built from frozen rules.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w+\b",
    )
    matrix = vectorizer.fit_transform([d["text"] for d in docs])

    preds = []

    for case in cases:
        cands = candidate_names(case)
        case_text = flatten_case_text(case)

        rec_score = {c: 0.0 for c in cands}
        avoid_score = {c: 0.0 for c in cands}
        caution_score = {c: 0.0 for c in cands}
        levels = {}
        reasoning = []
        sources = []

        for c in cands:
            q = vectorizer.transform([f"{case_text} candidate medication {c}"])
            sims = cosine_similarity(q, matrix)[0]
            top_idx = sims.argsort()[::-1][:10]

            for i in top_idx:
                score = float(sims[int(i)])
                if score < 0.03:
                    continue

                d = docs[int(i)]
                action = d["action"]
                target = d.get("target", "")

                if action in {"rec", "avoid", "caution"} and not loose_match(c, target):
                    continue

                if action == "rec":
                    rec_score[c] += score * 2.0
                    levels[c] = d.get("level") or "B"
                elif action == "avoid":
                    avoid_score[c] += score * 3.0
                elif action == "caution":
                    caution_score[c] += score * 2.0

                sources.append(d.get("source", "retrieval_doc"))
                reasoning.append(f"Retrieved {action} evidence for {c}: {d.get('doc_id')} score={score:.4f}")

        M_avoid = [c for c in cands if avoid_score[c] > 0]
        avoid_keys = {norm_text(x) for x in M_avoid}

        M_caution = [
            c for c in cands
            if caution_score[c] > 0 and norm_text(c) not in avoid_keys
        ]

        rec_pool = [
            c for c in cands
            if rec_score[c] > 0 and norm_text(c) not in avoid_keys
        ]
        rec_pool.sort(key=lambda c: rec_score[c] - caution_score[c], reverse=True)

        M_rec = rec_pool[:1]
        M_alt = alt_for_avoids(rules, M_avoid, allow_candidate_rec=M_rec)

        preds.append(
            make_prediction(
                case_id=case["case_id"],
                M_rec=M_rec,
                M_avoid=M_avoid,
                M_caution=M_caution,
                M_alt=M_alt,
                M_level={m: levels.get(m, "B") for m in M_rec},
                reasoning=reasoning or ["Retrieval-only engine found no confident evidence."],
                sources=unique_keep_order(sources) or ["retrieval_only_engine"],
                method="Retrieval-only Evidence Engine",
                trace_pass=True,
            )
        )

    return preds


def generic_kg(cases: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    krules = rules.get("knowledge_rules", []) or []
    edges = []

    for r in krules:
        effect = str(r.get("effect", "")).lower()
        if effect not in {"support", "avoid", "caution"}:
            continue

        for ctx in r.get("context_patterns", []) or ["older adult"]:
            for cand in r.get("candidate_patterns", []) or []:
                edges.append((ctx, effect, cand, r.get("source", r.get("id", "kg_edge"))))

    preds = []

    for case in cases:
        cands = candidate_names(case)
        case_text = flatten_case_text(case)

        scores = {
            c: {
                "support": 0,
                "avoid": 0,
                "caution": 0,
                "sources": [],
            }
            for c in cands
        }

        for ctx, effect, target, source in edges:
            if not loose_match(case_text, ctx):
                continue

            for c in cands:
                if not loose_match(c, target):
                    continue

                scores[c][effect] += 1
                scores[c]["sources"].append(source)

        M_avoid = [c for c in cands if scores[c]["avoid"] > 0]
        avoid_keys = {norm_text(x) for x in M_avoid}

        M_caution = [
            c for c in cands
            if scores[c]["caution"] > 0 and norm_text(c) not in avoid_keys
        ]

        rec_pool = [
            c for c in cands
            if scores[c]["support"] > 0 and norm_text(c) not in avoid_keys
        ]
        rec_pool.sort(key=lambda c: scores[c]["support"] - scores[c]["caution"], reverse=True)

        M_rec = rec_pool[:1]

        # Generic KG has no dedicated safer-alternative generation module.
        M_alt = []

        reasoning = [
            f"{c}: support={scores[c]['support']}, avoid={scores[c]['avoid']}, caution={scores[c]['caution']}"
            for c in cands
        ]

        sources = []
        for c in cands:
            sources.extend(scores[c]["sources"])

        preds.append(
            make_prediction(
                case_id=case["case_id"],
                M_rec=M_rec,
                M_avoid=M_avoid,
                M_caution=M_caution,
                M_alt=M_alt,
                M_level={m: "B" for m in M_rec},
                reasoning=reasoning,
                sources=unique_keep_order(sources) or ["generic_kg"],
                method="Generic KG Recommendation",
                trace_pass=True,
            )
        )

    return preds


METHODS = {
    "retrieval_only_engine": retrieval_only,
    "frozen_guideline": frozen_guideline,
    "generic_kg": generic_kg,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    args = parser.parse_args()

    cases = load_json(FINAL_INPUT)
    rules = load_json(FROZEN_RULES)

    preds = METHODS[args.method](cases, rules)

    out = OUT_DIR / f"final_predictions_{args.method}.json"
    save_json(preds, out)

    print(f"[OK] method={args.method}")
    print(f"[OK] saved={out}")
    print(f"[OK] predictions={len(preds)}")


if __name__ == "__main__":
    main()
