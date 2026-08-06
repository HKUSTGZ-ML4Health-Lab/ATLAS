#!/usr/bin/env python3
"""Verify development trajectories, policy freeze identity, and result invariance."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT = Path(__file__).resolve()
PD_ROOT = SCRIPT.parents[1]
DEV_ROOT = PD_ROOT.parent
CODE_APPENDIX = DEV_ROOT.parent

FORBIDDEN_DATA_TOKENS = (
    "final_test_gold",
    "test_gold",
    "offline_reference",
    "reference.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def string_literals(path: Path) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value


def validate_schema(record: Dict[str, Any], required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(record))
    if missing:
        raise AssertionError(f"Trajectory {record.get('trajectory_id')} is missing fields: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true", default=True)
    args = parser.parse_args()
    del args

    config = load_json(PD_ROOT / "teacher_config.json")
    schema = load_json(PD_ROOT / "trajectory_schema.json")
    guard = load_json(PD_ROOT / "freeze_guard.json")
    trajectories_path = PD_ROOT / "trajectories" / "dev39_k1_k2_k3.jsonl"
    candidate_path = PD_ROOT / "outputs" / "distilled_policy_candidate.json"
    manifest_path = PD_ROOT / "outputs" / "distillation_manifest.json"
    report_path = PD_ROOT / "outputs" / "stage1_verification.json"
    working_policy_path = DEV_ROOT / "rules" / "working_rules.json"
    frozen_policy_path = CODE_APPENDIX / "02_FROZEN_INFERENCE" / "frozen" / "frozen_rules.json"
    reported_inference_path = CODE_APPENDIX / "02_FROZEN_INFERENCE" / "scripts" / "run_final.py"

    records = read_jsonl(trajectories_path)
    expected_count = int(config["expected_trajectory_count"])
    if len(records) != expected_count:
        raise AssertionError(f"Expected {expected_count} trajectories, found {len(records)}")

    required = schema["required"]
    for record in records:
        validate_schema(record, required)

    dev_cases = load_json(DEV_ROOT / "data" / "dev_input.json")
    expected_cases = sorted(str(case["case_id"]) for case in dev_cases)
    expected_budgets = sorted(int(value) for value in config["teacher_review_budgets"])
    observed_pairs = sorted((str(record["development_case_id"]), int(record["teacher_review_budget"])) for record in records)
    expected_pairs = sorted((case_id, budget) for case_id in expected_cases for budget in expected_budgets)
    if observed_pairs != expected_pairs:
        raise AssertionError("Trajectory case-budget coverage is incomplete or duplicated")

    lower_trajectory_text = trajectories_path.read_text(encoding="utf-8").lower()
    bad_trajectory_tokens = [token for token in FORBIDDEN_DATA_TOKENS if token in lower_trajectory_text]
    if bad_trajectory_tokens:
        raise AssertionError(f"Forbidden evaluation-reference tokens found in trajectories: {bad_trajectory_tokens}")

    source_violations = []
    for source in sorted((PD_ROOT / "scripts").glob("*.py")):
        for literal in string_literals(source):
            lower = literal.lower()
            if any(token in lower for token in FORBIDDEN_DATA_TOKENS):
                if source.name != "verify_stage1_integrity.py":
                    source_violations.append((source.name, literal[:120]))
    if source_violations:
        raise AssertionError(f"Policy-distillation source reads or names evaluation references: {source_violations}")

    critical_hashes = {}
    for relative_path, expected_hash in guard["critical_files"].items():
        path = CODE_APPENDIX / relative_path
        observed_hash = sha256(path)
        critical_hashes[relative_path] = observed_hash
        if observed_hash != expected_hash:
            raise AssertionError(f"Critical file changed: {relative_path}")

    production_sources = list((CODE_APPENDIX / "02_FROZEN_INFERENCE").rglob("*.py")) + list((CODE_APPENDIX / "framework_src").rglob("*.py"))
    forbidden_imports = []
    for source in production_sources:
        text = source.read_text(encoding="utf-8", errors="ignore").lower()
        if "distilled_policy_candidate" in text or "policy_distillation" in text:
            forbidden_imports.append(str(source.relative_to(CODE_APPENDIX)))
    if forbidden_imports:
        raise AssertionError(f"Production inference contains an unexpected policy-distillation dependency: {forbidden_imports}")

    working_policy = load_json(working_policy_path)
    frozen_policy = load_json(frozen_policy_path)
    candidate = load_json(candidate_path)
    manifest = load_json(manifest_path)

    working_policy_sha256 = sha256(working_policy_path)
    frozen_policy_sha256 = sha256(frozen_policy_path)
    if working_policy_sha256 != frozen_policy_sha256:
        raise AssertionError("Development policy and frozen runtime policy are not byte-identical")
    if working_policy != frozen_policy:
        raise AssertionError("Development policy and frozen runtime policy differ as JSON values")
    if candidate.get("status") != "verified_policy_distillation_artifact":
        raise AssertionError("Released distillation artifact status is not verified")
    if candidate.get("runtime_policy_payload_field") != "base_policy_payload":
        raise AssertionError("Released distillation artifact does not identify its runtime policy payload")
    if candidate.get("runtime_policy_exact_match_verified") is not True:
        raise AssertionError("Released distillation artifact does not record exact policy verification")
    if candidate.get("base_policy_payload") != frozen_policy:
        raise AssertionError("Released distillation artifact does not contain the frozen runtime policy payload")
    if candidate.get("base_policy_sha256") != frozen_policy_sha256:
        raise AssertionError("Released distillation artifact policy hash does not match the frozen runtime policy")
    if candidate.get("runtime_policy_sha256") != frozen_policy_sha256:
        raise AssertionError("Released runtime policy hash does not match the frozen runtime policy")

    reported_literals = set(string_literals(reported_inference_path))
    if "frozen_rules.json" not in reported_literals or "frozen" not in reported_literals:
        raise AssertionError("Reported inference entry point does not load frozen/frozen_rules.json")

    freeze_verification = manifest.get("policy_freeze_verification") or {}
    expected_manifest_values = {
        "development_and_frozen_policy_byte_identical": True,
        "development_and_frozen_policy_json_identical": True,
        "distillation_artifact_policy_payload_exact_match": True,
        "reported_inference_loads_frozen_policy": True,
        "policy_sha256": frozen_policy_sha256,
    }
    for key, expected_value in expected_manifest_values.items():
        if freeze_verification.get(key) != expected_value:
            raise AssertionError(f"Manifest policy-freeze verification mismatch for {key}")

    report = {
        "schema_version": "atlas-policy-distillation-freeze-verification-v2",
        "status": "pass",
        "development_case_count": len(expected_cases),
        "teacher_review_budgets": expected_budgets,
        "trajectory_count": len(records),
        "case_budget_coverage_complete": True,
        "evaluation_reference_access": False,
        "production_policy_unchanged": True,
        "reported_experiment_artifacts_unchanged": True,
        "policy_freeze_verification": {
            "development_policy_path": "01_DEV39_WORKSPACE/rules/working_rules.json",
            "reported_runtime_policy_path": "02_FROZEN_INFERENCE/frozen/frozen_rules.json",
            "development_and_frozen_policy_byte_identical": True,
            "development_and_frozen_policy_json_identical": True,
            "distillation_artifact_policy_payload_exact_match": True,
            "reported_inference_loads_frozen_policy": True,
            "policy_sha256": frozen_policy_sha256,
        },
        "critical_file_sha256": critical_hashes,
        "trajectory_sha256": sha256(trajectories_path),
        "candidate_sha256": sha256(candidate_path),
        "manifest_sha256": sha256(manifest_path),
    }
    dump_json(report_path, report)
    print("[OK] Development trajectories and exact policy-freeze identity verified.")


if __name__ == "__main__":
    main()
