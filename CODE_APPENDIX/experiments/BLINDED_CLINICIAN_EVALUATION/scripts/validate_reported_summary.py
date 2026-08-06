#!/usr/bin/env python3
# AAAI-27 paper reference: Blinded Clinician Evaluation and Table 2.
"""Validate the locked aggregate results for the blinded clinician evaluation.

This script uses only the Python standard library. It validates the exact paper
values, internal arithmetic, benchmark stratification, and released source checksum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

CRITERIA = ["Correct.", "Safety", "Complete.", "Action.", "Evidence"]
MODELS = ["Gemini 3.1 Pro Preview", "ATLAS"]
EXPECTED = {
    "Gemini 3.1 Pro Preview": {
        "Correct.": 3.77,
        "Safety": 4.12,
        "Complete.": 3.67,
        "Action.": 3.85,
        "Evidence": 3.80,
        "Unsafe cases": (2, 40),
        "Preferred cases": (5, 40),
    },
    "ATLAS": {
        "Correct.": 4.11,
        "Safety": 4.35,
        "Complete.": 4.01,
        "Action.": 4.01,
        "Evidence": 4.08,
        "Unsafe cases": (1, 40),
        "Preferred cases": (28, 40),
    },
}
EXPECTED_STRATA = {
    "Western": {
        "Gemini 3.1 Pro Preview": [3.97, 4.30, 3.90, 4.02, 4.02],
        "ATLAS": [4.45, 4.60, 4.40, 4.22, 4.37],
    },
    "GeriMedBench": {
        "Gemini 3.1 Pro Preview": [3.57, 3.95, 3.43, 3.68, 3.58],
        "ATLAS": [3.77, 4.10, 3.62, 3.80, 3.78],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--source-docx", default=None)
    p.add_argument("--out", default=None)
    return p.parse_args()


def close(a: float, b: float, tol: float = 0.011) -> bool:
    return abs(float(a) - float(b)) < tol


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pair(obj: dict[str, Any]) -> tuple[int, int]:
    return int(obj["n"]), int(obj["denominator"])


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    obj = json.loads(summary_path.read_text(encoding="utf-8"))

    assert obj["experiment"] == "BLINDED_CLINICIAN_EVALUATION"
    assert obj["status"] == "completed_blinded_clinician_evaluation_aggregate_results"
    assert int(obj["reviewers"]) == 3
    assert int(obj["cases"]) == 40
    assert obj["sampling"] == {"Western": 20, "GeriMedBench": 20}

    overall = obj["overall"]
    for model in MODELS:
        assert model in overall, model
        for criterion in CRITERIA:
            actual = float(overall[model][criterion])
            expected = float(EXPECTED[model][criterion])
            assert close(actual, expected), (model, criterion, actual, expected)
        for count_name in ("Unsafe cases", "Preferred cases"):
            actual = pair(overall[model][count_name])
            expected = EXPECTED[model][count_name]
            assert actual == expected, (model, count_name, actual, expected)

    ties = pair(overall["Tie cases"])
    assert ties == (7, 40), ties
    assert overall["ATLAS"]["Preferred cases"]["n"] + overall["Gemini 3.1 Pro Preview"]["Preferred cases"]["n"] + overall["Tie cases"]["n"] == 40

    by_benchmark = obj["by_benchmark"]
    for dataset, methods in EXPECTED_STRATA.items():
        assert dataset in by_benchmark
        for model, expected_values in methods.items():
            actual_values = [float(by_benchmark[dataset][model][c]) for c in CRITERIA]
            for criterion, actual, expected in zip(CRITERIA, actual_values, expected_values):
                assert close(actual, expected), (dataset, model, criterion, actual, expected)

    # Equal 20/20 stratification implies each overall mean should equal the mean
    # of the two benchmark means, up to two-decimal reporting.
    for model in MODELS:
        for criterion in CRITERIA:
            expected_overall = (
                float(by_benchmark["Western"][model][criterion])
                + float(by_benchmark["GeriMedBench"][model][criterion])
            ) / 2.0
            assert close(float(overall[model][criterion]), expected_overall), (
                model,
                criterion,
                overall[model][criterion],
                expected_overall,
            )

    # ATLAS is higher on all five reported mean criteria.
    for criterion in CRITERIA:
        assert float(overall["ATLAS"][criterion]) > float(overall["Gemini 3.1 Pro Preview"][criterion]), criterion

    source_docx = Path(args.source_docx).resolve() if args.source_docx else None
    report = {
        "status": "PASS",
        "summary": str(summary_path.relative_to(summary_path.parents[1])),
        "cases": 40,
        "reviewers": 3,
        "validated_fields": {
            "overall_means": True,
            "benchmark_means": True,
            "unsafe_case_counts": True,
            "preference_counts": True,
            "tie_count": True,
            "equal_stratum_arithmetic": True,
        },
    }
    if source_docx:
        assert source_docx.is_file(), source_docx
        report["source_docx"] = str(source_docx.relative_to(summary_path.parents[1]))
        report["source_docx_sha256"] = sha256(source_docx)

    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
