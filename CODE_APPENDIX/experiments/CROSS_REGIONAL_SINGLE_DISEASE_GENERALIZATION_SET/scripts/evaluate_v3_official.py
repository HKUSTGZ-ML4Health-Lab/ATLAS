#!/usr/bin/env python3
"""
ATLAS V3 official offline evaluator wrapper.

Design:
- Reads predictions and gold only in this offline evaluation process.
- Imports framework_src/evaluation/metrics.py from the frozen ATLAS tree.
- Does not run inference, align predictions, rewrite slots, or inspect test input.
- Refuses to silently substitute a different evaluator in strict mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PRED_KEYS = {
    "pred", "preds", "prediction", "predictions", "prediction_data",
    "pred_data", "outputs", "results", "system_outputs",
}
GOLD_KEYS = {
    "gold", "golds", "reference", "references", "labels", "gold_data",
    "ground_truth", "targets", "annotations",
}
PRED_PATH_KEYS = {
    "pred_path", "prediction_path", "predictions_path", "pred_file",
    "prediction_file", "predictions_file",
}
GOLD_PATH_KEYS = {
    "gold_path", "reference_path", "references_path", "label_path",
    "labels_path", "gold_file", "reference_file",
}
OUTPUT_PATH_KEYS = {
    "out", "output", "output_path", "out_path", "result_path",
    "results_path", "save_path",
}

EXPLICIT_FUNCTION_PRIORITY = [
    "evaluate_predictions",
    "evaluate_dataset",
    "evaluate_all",
    "run_evaluation",
    "compute_all_metrics",
    "calculate_all_metrics",
    "aggregate_metrics",
    "evaluate",
    "compute_metrics",
    "calculate_metrics",
]

EXPLICIT_CLASS_PRIORITY = [
    "Evaluator",
    "MedicationSafetyEvaluator",
    "AtlasEvaluator",
    "ATLASMetrics",
    "MetricsEvaluator",
]


class EvaluationError(RuntimeError):
    pass


@dataclass
class Invocation:
    callable_name: str
    args: list[Any]
    kwargs: dict[str, Any]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Invalid JSON: {path}: {exc}") from exc


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def unwrap_records(obj: Any, kind: str) -> list[dict[str, Any]]:
    """Accept a list or a common top-level wrapper without changing record content."""
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict):
        preferred = (
            ["predictions", "results", "outputs", "data", "cases"]
            if kind == "pred"
            else ["gold", "labels", "references", "data", "cases"]
        )
        records = None
        for key in preferred:
            if isinstance(obj.get(key), list):
                records = obj[key]
                break
        if records is None:
            # Case-id keyed mapping.
            if obj and all(isinstance(v, dict) for v in obj.values()):
                records = []
                for key, value in obj.items():
                    item = dict(value)
                    item.setdefault("case_id", str(key))
                    records.append(item)
            else:
                raise EvaluationError(
                    f"Could not locate a record list in {kind} JSON. "
                    f"Top-level keys: {list(obj)[:30]}"
                )
    else:
        raise EvaluationError(f"{kind} JSON must be a list or object, got {type(obj).__name__}")

    if not all(isinstance(x, dict) for x in records):
        raise EvaluationError(f"Every {kind} record must be a JSON object.")
    return list(records)


def record_case_id(record: Mapping[str, Any]) -> str | None:
    value = record.get("case_id")
    return None if value is None else str(value)


def validate_case_alignment(pred_records: list[dict[str, Any]],
                            gold_records: list[dict[str, Any]]) -> dict[str, Any]:
    pred_ids = [record_case_id(x) for x in pred_records]
    gold_ids = [record_case_id(x) for x in gold_records]

    if any(x is None for x in pred_ids):
        raise EvaluationError("At least one prediction is missing case_id.")
    if any(x is None for x in gold_ids):
        raise EvaluationError("At least one gold record is missing case_id.")
    if len(set(pred_ids)) != len(pred_ids):
        raise EvaluationError("Duplicate case_id detected in predictions.")
    if len(set(gold_ids)) != len(gold_ids):
        raise EvaluationError("Duplicate case_id detected in gold.")

    pred_set = set(pred_ids)
    gold_set = set(gold_ids)
    missing = sorted(gold_set - pred_set)
    extra = sorted(pred_set - gold_set)
    if missing or extra:
        raise EvaluationError(
            "Prediction/gold case IDs do not match. "
            f"missing_predictions={missing[:20]}, extra_predictions={extra[:20]}"
        )

    return {
        "N_predictions": len(pred_records),
        "N_gold": len(gold_records),
        "same_order": pred_ids == gold_ids,
        "case_id_sets_equal": True,
    }


def import_metrics(metrics_path: Path):
    """
    Import the official metrics module with package context preserved.

    metrics.py contains relative imports such as:
        from ..system_impl.metrics_impl import *
    Therefore it must be imported as framework_src.evaluation.metrics rather
    than loaded as an isolated file.
    """
    metrics_path = metrics_path.resolve()

    atlas_root = None
    package_name = None

    parts = metrics_path.parts
    if "framework_src" in parts:
        idx = parts.index("framework_src")
        atlas_root = Path(*parts[:idx]) if idx > 0 else Path("/")
        relative = metrics_path.relative_to(atlas_root).with_suffix("")
        package_name = ".".join(relative.parts)

    if atlas_root is None or package_name is None:
        raise EvaluationError(
            f"Could not infer the ATLAS package root from metrics path: {metrics_path}. "
            "Expected a path such as <ATLAS>/framework_src/evaluation/metrics.py"
        )

    atlas_root_str = str(atlas_root)
    if atlas_root_str not in sys.path:
        sys.path.insert(0, atlas_root_str)

    try:
        import importlib
        return importlib.import_module(package_name)
    except Exception as exc:
        raise EvaluationError(
            f"Failed to import official metrics as package '{package_name}' "
            f"with ATLAS root '{atlas_root}': {type(exc).__name__}: {exc}"
        ) from exc


def annotation_prefers_path(annotation: Any) -> bool:
    if annotation is inspect.Signature.empty:
        return False
    text = str(annotation).lower()
    return "path" in text or "str" in text or "os.pathlike" in text


def choose_value_for_parameter(
    param: inspect.Parameter,
    pred_raw: Any,
    gold_raw: Any,
    pred_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    pred_path: Path,
    gold_path: Path,
    out_path: Path,
) -> tuple[bool, Any]:
    name = param.name.lower()

    if name in PRED_PATH_KEYS:
        return True, pred_path if annotation_prefers_path(param.annotation) else str(pred_path)
    if name in GOLD_PATH_KEYS:
        return True, gold_path if annotation_prefers_path(param.annotation) else str(gold_path)
    if name in OUTPUT_PATH_KEYS:
        return True, out_path if annotation_prefers_path(param.annotation) else str(out_path)

    if name in PRED_KEYS or any(token in name for token in ("prediction", "predictions")):
        if "path" in name or "file" in name:
            return True, str(pred_path)
        return True, pred_records

    if name in GOLD_KEYS or any(token in name for token in ("ground_truth", "reference", "gold")):
        if "path" in name or "file" in name:
            return True, str(gold_path)
        return True, gold_records

    # Some official modules call dataset-level inputs y_pred/y_true.
    if name in {"y_pred", "ypred"}:
        return True, pred_records
    if name in {"y_true", "ytrue"}:
        return True, gold_records

    # Raw top-level containers are available when the function explicitly requests them.
    if name in {"pred_raw", "predictions_raw"}:
        return True, pred_raw
    if name in {"gold_raw", "references_raw"}:
        return True, gold_raw

    return False, None


def build_invocation(
    fn: Callable[..., Any],
    pred_raw: Any,
    gold_raw: Any,
    pred_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    pred_path: Path,
    gold_path: Path,
    out_path: Path,
) -> Invocation | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    data_params = 0

    for param in sig.parameters.values():
        if param.name in {"self", "cls"}:
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        found, value = choose_value_for_parameter(
            param, pred_raw, gold_raw, pred_records, gold_records,
            pred_path, gold_path, out_path,
        )
        if found:
            lower = param.name.lower()
            if lower in PRED_KEYS | GOLD_KEYS | PRED_PATH_KEYS | GOLD_PATH_KEYS or any(
                token in lower for token in ("pred", "gold", "reference", "ground_truth")
            ):
                data_params += 1
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                args.append(value)
            else:
                kwargs[param.name] = value
        elif param.default is inspect.Signature.empty:
            return None

    # Dataset evaluator should consume both predictions and references, or paths to both.
    if data_params < 2:
        return None
    return Invocation(getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn))), args, kwargs)


def dict_from_result(result: Any) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict") and callable(result.to_dict):
        converted = result.to_dict()
        if isinstance(converted, dict):
            return converted
    if isinstance(result, (tuple, list)):
        for item in result:
            converted = dict_from_result(item)
            if converted is not None:
                return converted
    return None


def looks_like_evaluation_result(result: Mapping[str, Any]) -> bool:
    keys = {str(k).lower() for k in result.keys()}
    if "summary" in keys:
        return True
    markers = {
        "total_cases", "n", "strict_success_count", "success_rate_strict",
        "micro", "unsafe_recommendation_rate", "trace_consistency_pass_rate",
        "overall_safety_reasoning", "osrs",
    }
    return len(keys & markers) >= 2


def public_function_candidates(module) -> list[tuple[str, Callable[..., Any]]]:
    found: dict[str, Callable[..., Any]] = {}
    for name in EXPLICIT_FUNCTION_PRIORITY:
        obj = getattr(module, name, None)
        if inspect.isfunction(obj):
            found[name] = obj

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        lower = name.lower()
        if any(token in lower for token in ("evaluate", "all_metrics", "aggregate", "summary")):
            found.setdefault(name, obj)

    priority = {name: i for i, name in enumerate(EXPLICIT_FUNCTION_PRIORITY)}
    return sorted(found.items(), key=lambda x: (priority.get(x[0], 10_000), x[0]))


def class_method_candidates(module) -> list[tuple[str, Any, Callable[..., Any]]]:
    classes: list[tuple[str, type]] = []
    for name in EXPLICIT_CLASS_PRIORITY:
        obj = getattr(module, name, None)
        if inspect.isclass(obj):
            classes.append((name, obj))

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.startswith("_") or obj.__module__ != module.__name__:
            continue
        if any(token in name.lower() for token in ("evaluat", "metric", "score")):
            if all(name != seen for seen, _ in classes):
                classes.append((name, obj))

    methods: list[tuple[str, Any, Callable[..., Any]]] = []
    for class_name, cls in classes:
        try:
            ctor_sig = inspect.signature(cls)
            required = [
                p for p in ctor_sig.parameters.values()
                if p.default is inspect.Signature.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                   inspect.Parameter.VAR_KEYWORD)
            ]
            if required:
                continue
            instance = cls()
        except Exception:
            continue

        for method_name in EXPLICIT_FUNCTION_PRIORITY:
            method = getattr(instance, method_name, None)
            if callable(method):
                methods.append((f"{class_name}.{method_name}", instance, method))
        for method_name, method in inspect.getmembers(instance, callable):
            if method_name.startswith("_"):
                continue
            if any(token in method_name.lower() for token in ("evaluate", "all_metrics", "summary")):
                label = f"{class_name}.{method_name}"
                if all(label != seen for seen, _, _ in methods):
                    methods.append((label, instance, method))
    return methods


def callable_inventory(module) -> list[dict[str, str]]:
    inventory = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.isfunction(obj) or inspect.isclass(obj):
            try:
                signature = str(inspect.signature(obj))
            except Exception:
                signature = "<unknown>"
            inventory.append({
                "name": name,
                "kind": "class" if inspect.isclass(obj) else "function",
                "signature": signature,
            })
    return inventory


def run_official_api(
    module,
    pred_raw: Any,
    gold_raw: Any,
    pred_records: list[dict[str, Any]],
    gold_records: list[dict[str, Any]],
    pred_path: Path,
    gold_path: Path,
    out_path: Path,
) -> tuple[dict[str, Any], str, list[dict[str, str]]]:
    attempts: list[dict[str, str]] = []

    candidates: list[tuple[str, Callable[..., Any]]] = public_function_candidates(module)
    candidates.extend((label, method) for label, _, method in class_method_candidates(module))

    for label, fn in candidates:
        invocation = build_invocation(
            fn, pred_raw, gold_raw, pred_records, gold_records,
            pred_path, gold_path, out_path,
        )
        if invocation is None:
            attempts.append({"callable": label, "status": "signature_not_compatible"})
            continue

        try:
            result = fn(*invocation.args, **invocation.kwargs)
            result_dict = dict_from_result(result)

            # Some official functions save the result and return None.
            if result_dict is None and out_path.exists():
                loaded = read_json(out_path)
                if isinstance(loaded, dict):
                    result_dict = loaded

            if result_dict is None:
                attempts.append({"callable": label, "status": "no_dictionary_result"})
                continue
            if not looks_like_evaluation_result(result_dict):
                attempts.append({"callable": label, "status": "result_not_dataset_summary"})
                continue

            attempts.append({"callable": label, "status": "success"})
            return result_dict, label, attempts
        except Exception as exc:
            attempts.append({
                "callable": label,
                "status": "raised_exception",
                "error": f"{type(exc).__name__}: {exc}",
            })

    inventory = callable_inventory(module)
    raise EvaluationError(
        "No compatible dataset-level official evaluator API was found in metrics.py.\n"
        "This wrapper deliberately did not substitute a different metric implementation.\n"
        "Callable inventory:\n" + json.dumps(inventory, ensure_ascii=False, indent=2) +
        "\nAttempts:\n" + json.dumps(attempts, ensure_ascii=False, indent=2)
    )


def compact_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", result)
    micro = summary.get("micro", {}) if isinstance(summary, dict) else {}

    def dig(*keys: str) -> Any:
        value: Any = summary
        for key in keys:
            if not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return value

    def pct(value: Any) -> Any:
        if value is None:
            return None
        number = float(value)
        # Official full evaluator commonly stores fractions in [0,1].
        if 0.0 <= number <= 1.0:
            number *= 100.0
        return round(number, 2)

    osrs = None
    overall = dig("overall_safety_reasoning")
    if isinstance(overall, Mapping):
        osrs = overall.get("overall_safety_reasoning_score_percent")
        if osrs is None:
            osrs = overall.get("OSRS", overall.get("osrs"))
    if osrs is None:
        osrs = dig("OSRS")
    if osrs is None:
        osrs = dig("osrs")

    compact = {
        "N": dig("total_cases") if dig("total_cases") is not None else dig("N"),
        "strict_success_count": dig("strict_success_count"),
        "strict_failed_count": dig("strict_failed_count")
            if dig("strict_failed_count") is not None else dig("failed_cases"),
        "success_rate_strict": pct(dig("success_rate_strict")),
        "M_rec_f1": pct(micro.get("M_rec", {}).get("f1"))
            if isinstance(micro.get("M_rec", {}), Mapping) else None,
        "M_avoid_recall": pct(micro.get("M_avoid", {}).get("recall"))
            if isinstance(micro.get("M_avoid", {}), Mapping) else None,
        "M_avoid_f1": pct(micro.get("M_avoid", {}).get("f1"))
            if isinstance(micro.get("M_avoid", {}), Mapping) else None,
        "M_caution_f1": pct(micro.get("M_caution", {}).get("f1"))
            if isinstance(micro.get("M_caution", {}), Mapping) else None,
        "M_alt_f1": pct(micro.get("M_alt", {}).get("f1"))
            if isinstance(micro.get("M_alt", {}), Mapping) else None,
        "unsafe_rate": pct(
            dig("unsafe_recommendation_rate")
            if dig("unsafe_recommendation_rate") is not None
            else dig("unsafe_rate")
        ),
        "trace_pass_rate": pct(
            dig("trace_consistency_pass_rate")
            if dig("trace_consistency_pass_rate") is not None
            else dig("trace_pass_rate")
        ),
        "OSRS": pct(osrs),
    }

    # Some official APIs already return a compact summary.
    aliases = {
        "M_rec_f1": ["M_rec_f1"],
        "M_avoid_recall": ["M_avoid_recall"],
        "M_avoid_f1": ["M_avoid_f1"],
        "M_caution_f1": ["M_caution_f1"],
        "M_alt_f1": ["M_alt_f1"],
    }
    for target, source_keys in aliases.items():
        if compact[target] is None:
            for key in source_keys:
                if isinstance(summary, Mapping) and summary.get(key) is not None:
                    compact[target] = pct(summary[key])
                    break
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline wrapper for ATLAS framework_src/evaluation/metrics.py"
    )
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("framework_src/evaluation/metrics.py"),
    )
    parser.add_argument(
        "--compact-out",
        type=Path,
        default=None,
        help="Optional compact AAAI-table summary JSON.",
    )
    args = parser.parse_args()

    pred_path = args.pred.resolve()
    gold_path = args.gold.resolve()
    out_path = args.out.resolve()
    metrics_path = args.metrics.resolve()
    compact_path = (
        args.compact_out.resolve()
        if args.compact_out is not None
        else out_path.with_name(out_path.stem + "_compact.json")
    )

    for label, path in [
        ("prediction", pred_path),
        ("gold", gold_path),
        ("official metrics", metrics_path),
    ]:
        if not path.is_file():
            raise EvaluationError(f"Missing {label} file: {path}")

    # Offline boundary: gold is loaded only inside this evaluator process.
    pred_raw = read_json(pred_path)
    gold_raw = read_json(gold_path)
    pred_records = unwrap_records(pred_raw, "pred")
    gold_records = unwrap_records(gold_raw, "gold")
    alignment = validate_case_alignment(pred_records, gold_records)

    module = import_metrics(metrics_path)
    result, official_callable, attempts = run_official_api(
        module=module,
        pred_raw=pred_raw,
        gold_raw=gold_raw,
        pred_records=pred_records,
        gold_records=gold_records,
        pred_path=pred_path,
        gold_path=gold_path,
        out_path=out_path,
    )

    provenance = {
        "evaluator_wrapper": "ATLAS_V3_OFFICIAL_EVALUATOR_V1",
        "official_metrics_path": str(metrics_path),
        "official_metrics_sha256": sha256_file(metrics_path),
        "official_callable": official_callable,
        "predictions_path": str(pred_path),
        "predictions_sha256": sha256_file(pred_path),
        "gold_path": str(gold_path),
        "gold_sha256": sha256_file(gold_path),
        "alignment": alignment,
        "gold_used_during_inference": False,
        "gold_used_by_offline_evaluator_only": True,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "call_attempts": attempts,
    }

    final_result = dict(result)
    existing_provenance = final_result.get("evaluation_provenance")
    if isinstance(existing_provenance, Mapping):
        final_result["evaluation_provenance"] = {
            **dict(existing_provenance),
            **provenance,
        }
    else:
        final_result["evaluation_provenance"] = provenance

    write_json(out_path, final_result)
    compact = compact_summary(final_result)
    compact["evaluation_provenance"] = provenance
    write_json(compact_path, compact)

    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"\n[OK] Full official result: {out_path}")
    print(f"[OK] Compact summary:     {compact_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvaluationError as exc:
        print(f"[EVALUATION ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception:
        traceback.print_exc()
        raise SystemExit(3)
