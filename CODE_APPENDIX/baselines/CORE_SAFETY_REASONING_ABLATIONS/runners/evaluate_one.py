from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from framework_src.evaluation.metrics import evaluate
from baselines_v2.common import RESULT_DIR, load_json, save_json


ROOT = Path(__file__).resolve().parents[2]
FINAL_GOLD = ROOT / "03_OFFLINE_EVALUATION" / "gold" / "final_test_gold.json"


def pct(x):
    return round(float(x) * 100.0, 2)


def flatten_summary(summary):
    macro = summary.get("macro", {}) or {}
    osrs = summary.get("overall_safety_reasoning", {}) or {}

    return {
        "N": summary.get("total_cases", 0),
        "strict_success_count": summary.get("success_cases", 0),
        "strict_failed_count": summary.get("failed_cases", 0),
        "success_rate_strict": pct(summary.get("success_rate_strict", 0.0)),
        "M_rec_f1": pct(macro.get("M_rec_f1", 0.0)),
        "M_avoid_recall": pct(macro.get("M_avoid_recall", 0.0)),
        "M_avoid_f1": pct(macro.get("M_avoid_f1", 0.0)),
        "M_caution_f1": pct(macro.get("M_caution_f1", 0.0)),
        "M_alt_f1": pct(macro.get("M_alt_f1", 0.0)),
        "unsafe_rate": pct(summary.get("unsafe_recommendation_rate", 0.0)),
        "trace_pass_rate": pct(summary.get("trace_consistency_pass_rate", 0.0)),
        "OSRS": round(float(osrs.get("overall_safety_reasoning_score_percent", 0.0)), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()

    predictions = load_json(args.pred)
    gold_rows = load_json(FINAL_GOLD)

    raw = evaluate(predictions, gold_rows)
    summary = raw["summary"]
    flat = flatten_summary(summary)

    save_json(raw, RESULT_DIR / f"eval_{args.name}.json")
    save_json(flat, RESULT_DIR / f"summary_{args.name}.json")

    print(json.dumps(flat, ensure_ascii=False, indent=2))
    print(f"[OK] saved {RESULT_DIR / f'eval_{args.name}.json'}")
    print(f"[OK] saved {RESULT_DIR / f'summary_{args.name}.json'}")


if __name__ == "__main__":
    main()
