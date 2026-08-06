#!/usr/bin/env python3
# AAAI-27 paper reference: Blinded Clinician Evaluation and Table 2.
"""Compare a raw-rating aggregation result with the locked reported summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MAP = {
    "Correct.": "correctness",
    "Safety": "safety",
    "Complete.": "completeness",
    "Action.": "actionability",
    "Evidence": "evidence_consistency",
}
MODELS = ["Gemini 3.1 Pro Preview", "ATLAS"]


def close(a: float, b: float, tol: float = 0.011) -> bool:
    return abs(float(a) - float(b)) < tol


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aggregated", required=True)
    p.add_argument("--reported", required=True)
    args = p.parse_args()
    agg = json.loads(Path(args.aggregated).read_text(encoding="utf-8"))
    rep = json.loads(Path(args.reported).read_text(encoding="utf-8"))
    errors = []
    for model in MODELS:
        for paper_name, raw_name in MAP.items():
            a = agg["methods"][model][raw_name]["mean"]
            r = rep["overall"][model][paper_name]
            if not close(a, r):
                errors.append((model, paper_name, a, r))
        a_unsafe = agg["methods"][model]["unsafe_cases"]["n"]
        r_unsafe = rep["overall"][model]["Unsafe cases"]["n"]
        if int(a_unsafe) != int(r_unsafe):
            errors.append((model, "Unsafe cases", a_unsafe, r_unsafe))
        a_pref = agg["preference"][model]["n"]
        r_pref = rep["overall"][model]["Preferred cases"]["n"]
        if int(a_pref) != int(r_pref):
            errors.append((model, "Preferred cases", a_pref, r_pref))
    a_ties = agg["preference"]["Tie"]["n"]
    r_ties = rep["overall"]["Tie cases"]["n"]
    if int(a_ties) != int(r_ties):
        errors.append(("Tie", "cases", a_ties, r_ties))
    if errors:
        raise AssertionError(errors)
    print("[OK] Raw-rating aggregation matches the locked reported summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
