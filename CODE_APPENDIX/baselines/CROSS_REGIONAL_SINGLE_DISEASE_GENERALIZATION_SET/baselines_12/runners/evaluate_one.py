from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASELINE_ROOT = Path(os.environ.get("ATLAS_SD_BASELINE_ROOT", str(Path(__file__).resolve().parents[2]))).resolve()
ATLAS_ROOT = Path(os.environ.get("ATLAS_ROOT", str(Path(__file__).resolve().parents[4]))).expanduser().resolve()
sys.path.insert(0, str(ATLAS_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
from framework_src.evaluation.metrics import evaluate
from baselines_12.common import RESULT_DIR, load_json, save_json
REFERENCE = BASELINE_ROOT / "offline_reference" / "ATLAS_SINGLE_DISEASE_GENERALIZATION_reference.json"


def pct(x): return round(float(x or 0.0)*100.0,2)

def flatten(summary):
    macro=summary.get("macro",{}) or {}; osrs=summary.get("overall_safety_reasoning",{}) or {}
    n=int(summary.get("total_cases",0) or 0); failed=int(summary.get("failed_cases",0) or 0)
    success=summary.get("success_cases")
    if success is None: success=n-failed
    return {"N":n,"strict_success_count":int(success),"strict_failed_count":failed,"success_rate_strict":pct(summary.get("success_rate_strict",0.0)),"M_rec_f1":pct(macro.get("M_rec_f1",0.0)),"M_avoid_recall":pct(macro.get("M_avoid_recall",0.0)),"M_avoid_f1":pct(macro.get("M_avoid_f1",0.0)),"M_caution_f1":pct(macro.get("M_caution_f1",0.0)),"M_alt_f1":pct(macro.get("M_alt_f1",0.0)),"unsafe_rate":pct(summary.get("unsafe_recommendation_rate",0.0)),"trace_pass_rate":pct(summary.get("trace_consistency_pass_rate",0.0)),"OSRS":round(float(osrs.get("overall_safety_reasoning_score_percent",0.0)),2)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--name",required=True); p.add_argument("--pred",required=True); args=p.parse_args()
    predictions=load_json(args.pred); reference=load_json(REFERENCE)
    if len(predictions)!=612 or len(reference)!=612: raise RuntimeError(f"Expected 612 predictions/reference, got {len(predictions)}/{len(reference)}")
    raw=evaluate(predictions,reference); flat=flatten(raw["summary"])
    save_json(raw,RESULT_DIR/f"eval_{args.name}.json"); save_json(flat,RESULT_DIR/f"summary_{args.name}.json")
    print(json.dumps(flat,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
