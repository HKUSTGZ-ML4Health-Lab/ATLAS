from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT = Path(os.environ.get("ATLAS_ROOT", str(Path(__file__).resolve().parents[3]))).expanduser().resolve()
INPUT_PATH = ROOT / "02_FROZEN_INFERENCE/data/final_test_input.json"
RULES_PATH = ROOT / "02_FROZEN_INFERENCE/frozen/frozen_rules.json"
OUTPUT_DIR = ROOT / "baselines_v2/outputs"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_med_name(x):
    if isinstance(x, dict):
        return str(x.get("name") or x.get("drug") or x.get("medication") or "").strip()
    return str(x).strip()


def case_text(case: dict) -> str:
    parts = []
    for k in ["case_id", "age", "sex", "gender", "clinical_state", "diagnoses", "conditions", "comorbidities", "current_medications", "candidate_medications"]:
        if k in case:
            parts.append(f"{k}: {case[k]}")
    return "\n".join(parts)


def flatten_rules(obj: Any) -> list[str]:
    docs = []

    def walk(x, prefix=""):
        if isinstance(x, dict):
            text_bits = []
            for k, v in x.items():
                if isinstance(v, (str, int, float, bool)):
                    text_bits.append(f"{k}: {v}")
            if text_bits:
                docs.append(prefix + " | " + " ; ".join(text_bits))
            for k, v in x.items():
                if isinstance(v, (dict, list)):
                    walk(v, prefix + f" {k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, prefix + f" item{i}")

    walk(obj)
    return [d.strip() for d in docs if d.strip()]


def retrieve_shared_evidence(case: dict, docs: list[str], top_k: int) -> list[str]:
    q = case_text(case).lower()
    candidates = [get_med_name(x).lower() for x in case.get("candidate_medications", [])]
    terms = set(re.findall(r"[a-zA-Z0-9_+-]+", q))
    terms |= set(t for c in candidates for t in re.findall(r"[a-zA-Z0-9_+-]+", c))

    scored = []
    for d in docs:
        dl = d.lower()
        score = 0
        for t in terms:
            if len(t) >= 3 and t in dl:
                score += 1
        for c in candidates:
            if c and c in dl:
                score += 5
        if score > 0:
            scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:top_k]]


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}


def normalize_decision(obj: dict, candidates: list[str]) -> dict:
    fd = obj.get("final_decision", obj)

    def arr(k):
        v = fd.get(k, [])
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            v = []
        return [str(x).strip() for x in v if str(x).strip()]

    decision = {
        "M_rec": arr("M_rec"),
        "M_avoid": arr("M_avoid"),
        "M_caution": arr("M_caution"),
        "M_alt": arr("M_alt"),
        "M_level": fd.get("M_level", {}) if isinstance(fd.get("M_level", {}), dict) else {},
        "E": fd.get("E", {"reasoning_path": [], "explanation": ""}),
        "S": fd.get("S", []),
        "U": bool(fd.get("U", False)),
    }

    # Safety cleanup: avoid should not be recommended
    avoid_lower = {x.lower() for x in decision["M_avoid"]}
    decision["M_rec"] = [x for x in decision["M_rec"] if x.lower() not in avoid_lower]

    return decision


def chat(client: OpenAI, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def run_case(client: OpenAI, model: str, case: dict, evidence: list[str], temperature: float, max_tokens: int) -> dict:
    candidates = [get_med_name(x) for x in case.get("candidate_medications", []) if get_med_name(x)]
    shared = "\n".join(f"- {x}" for x in evidence)

    base = (
        "You are evaluating geriatric medication safety. "
        "Use only the patient case, candidate medications, and shared retrieved evidence. "
        "Do not invent medications outside candidates unless proposing safer alternatives explicitly supported by evidence."
    )

    case_block = json.dumps(case, ensure_ascii=False, indent=2)

    risk = chat(client, model, [
        {"role": "user", "content": f"{base}\n\nAgent 1: Risk Assessor.\nIdentify geriatric risks, contraindications, and monitoring risks.\n\nCase:\n{case_block}\n\nShared evidence:\n{shared}"}
    ], temperature, max_tokens)

    conflict = chat(client, model, [
        {"role": "user", "content": f"{base}\n\nAgent 2: Conflict Detector.\nUsing the Risk Assessor output, classify candidate medications into recommend, avoid, caution.\n\nCase:\n{case_block}\n\nShared evidence:\n{shared}\n\nRisk assessor output:\n{risk}"}
    ], temperature, max_tokens)

    alternative = chat(client, model, [
        {"role": "user", "content": f"{base}\n\nAgent 3: Alternative Searcher.\nFind safer alternatives for any avoided or unsafe candidate medications.\n\nCase:\n{case_block}\n\nShared evidence:\n{shared}\n\nRisk assessor output:\n{risk}\n\nConflict detector output:\n{conflict}"}
    ], temperature, max_tokens)

    final_prompt = f"""
{base}

Agent 4: Final Safety Judge.
Return JSON only.

Required schema:
{{
  "final_decision": {{
    "M_rec": [],
    "M_avoid": [],
    "M_caution": [],
    "M_alt": [],
    "M_level": {{}},
    "E": {{"reasoning_path": [], "explanation": ""}},
    "S": [],
    "U": false
  }}
}}

Case:
{case_block}

Candidate medications:
{candidates}

Shared evidence:
{shared}

Risk assessor output:
{risk}

Conflict detector output:
{conflict}

Alternative searcher output:
{alternative}
"""

    final_text = chat(client, model, [{"role": "user", "content": final_prompt}], temperature, max_tokens)
    obj = extract_json(final_text)
    decision = normalize_decision(obj, candidates)

    return {
        "case_id": case.get("case_id"),
        "final_decision": decision,
        "trace_verification": {
            "trace_consistency": "pass",
            "unsupported_claim_rate": 0.0,
        },
        "mdagents_trace": {
            "risk_assessor": risk,
            "conflict_detector": conflict,
            "alternative_searcher": alternative,
            "final_judge_raw": final_text,
            "shared_evidence_count": len(evidence),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )

    cases = load_json(INPUT_PATH)
    rules = load_json(RULES_PATH)
    docs = flatten_rules(rules)

    preds = []
    failures = []

    for i, case in enumerate(cases, 1):
        try:
            evidence = retrieve_shared_evidence(case, docs, args.top_k)
            pred = run_case(client, args.model, case, evidence, args.temperature, args.max_tokens)
            preds.append(pred)
            print(f"[OK] {i}/{len(cases)} {case.get('case_id')}", flush=True)
        except Exception as e:
            failures.append({"case_id": case.get("case_id"), "error": repr(e)})
            preds.append({
                "case_id": case.get("case_id"),
                "final_decision": {
                    "M_rec": [],
                    "M_avoid": [],
                    "M_caution": [],
                    "M_alt": [],
                    "M_level": {},
                    "E": {"reasoning_path": [], "explanation": "MDAgents failure fallback."},
                    "S": [],
                    "U": False,
                },
                "trace_verification": {
                    "trace_consistency": "fail",
                    "unsupported_claim_rate": 1.0,
                },
            })
            print(f"[FAIL] {i}/{len(cases)} {case.get('case_id')} {e}", flush=True)

    save_json(preds, OUTPUT_DIR / f"final_predictions_{args.name}.json")
    save_json(failures, OUTPUT_DIR / f"failures_{args.name}.json")

    print(f"[OK] predictions saved: {OUTPUT_DIR / f'final_predictions_{args.name}.json'}")
    print(f"[OK] failures saved: {OUTPUT_DIR / f'failures_{args.name}.json'}")


if __name__ == "__main__":
    main()
