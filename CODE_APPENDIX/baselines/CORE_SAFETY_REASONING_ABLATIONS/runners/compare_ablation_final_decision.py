from __future__ import annotations

import csv
import json
from pathlib import Path

FIELDS = ["M_rec", "M_avoid", "M_caution", "M_alt", "U"]

VARIANTS = [
    "ablation_no_pmcg",
    "ablation_no_drug_conflict_auditor",
    "ablation_no_geriatric_risk_auditor",
    "ablation_no_safety_gate",
    "ablation_no_alternative_agent",
    "ablation_no_safety_critic_revision",
    "ablation_no_open_candidate_generation",
    "ablation_no_trace_verifier",
]

OUT = Path("baselines_v2/results/case_level_ablation_final_decision_diff.csv")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def norm_item(x):
    return str(x).strip().lower()


def norm_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return sorted(norm_item(i) for i in x)
    return [norm_item(x)]


def extract_final_decision(pred):
    d = pred.get("final_decision")
    if not isinstance(d, dict):
        raise ValueError(f"Missing final_decision for case_id={pred.get('case_id')}")

    return {
        "M_rec": norm_list(d.get("M_rec")),
        "M_avoid": norm_list(d.get("M_avoid")),
        "M_caution": norm_list(d.get("M_caution")),
        "M_alt": norm_list(d.get("M_alt")),
        "U": d.get("U"),
    }


def main():
    full = load("baselines_v2/outputs/final_predictions_atlas.json")
    full_map = {
        str(x.get("case_id")): extract_final_decision(x)
        for x in full
    }

    rows = []

    for variant in VARIANTS:
        data = load(f"baselines_v2/outputs/final_predictions_{variant}.json")

        changed_cases = 0
        changed_by_field = {f: 0 for f in FIELDS}
        examples = []

        for x in data:
            cid = str(x.get("case_id"))
            fd = full_map[cid]
            ad = extract_final_decision(x)

            case_changed = False

            for f in FIELDS:
                if fd[f] != ad[f]:
                    changed_by_field[f] += 1
                    case_changed = True

            if case_changed:
                changed_cases += 1

                if len(examples) < 5:
                    examples.append({
                        "case_id": cid,
                        "full": fd,
                        "ablation": ad,
                    })

        row = {
            "variant": variant,
            "changed_cases": changed_cases,
            "total_cases": 201,
            "changed_M_rec": changed_by_field["M_rec"],
            "changed_M_avoid": changed_by_field["M_avoid"],
            "changed_M_caution": changed_by_field["M_caution"],
            "changed_M_alt": changed_by_field["M_alt"],
            "changed_U": changed_by_field["U"],
        }

        rows.append(row)

        print()
        print("=" * 100)
        print(variant)
        print("changed_cases:", changed_cases, "/ 201")
        print("changed_by_field:", changed_by_field)

        if examples:
            print("examples:")
            for e in examples:
                print("case_id:", e["case_id"])
                print("  full    :", e["full"])
                print("  ablation:", e["ablation"])
        else:
            print("examples: none")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("[OK] saved:", OUT)


if __name__ == "__main__":
    main()
