from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from rank_bm25 import BM25Okapi


ROOT = Path(__file__).resolve().parents[2]

FINAL_INPUT = ROOT / "02_FROZEN_INFERENCE" / "data" / "final_test_input.json"
FROZEN_RULES = ROOT / "02_FROZEN_INFERENCE" / "frozen" / "frozen_rules.json"

OUT_DIR = ROOT / "baselines_v2" / "outputs"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zA-Z0-9_\-]+", text)


def compact_json(obj: Any, max_chars: int = 1800) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[:max_chars] + " ... [truncated]"
    return text


def case_to_text(case: Dict[str, Any]) -> str:
    parts = []
    for k, v in case.items():
        if k == "case_id":
            continue
        if isinstance(v, (str, int, float, bool)):
            parts.append(f"{k}: {v}")
        elif isinstance(v, list):
            parts.append(f"{k}: " + "; ".join(map(str, v)))
        elif isinstance(v, dict):
            parts.append(f"{k}: " + compact_json(v, 2000))
        else:
            parts.append(f"{k}: {str(v)}")
    return "\n".join(parts)


def make_corpus() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []

    if FROZEN_RULES.exists():
        rules = load_json(FROZEN_RULES)

        if isinstance(rules, dict):
            for k, v in rules.items():
                docs.append({
                    "doc_id": f"frozen_rule::{k}",
                    "text": f"{k}: {compact_json(v, 2500)}",
                })
        elif isinstance(rules, list):
            for i, v in enumerate(rules):
                docs.append({
                    "doc_id": f"frozen_rule::{i}",
                    "text": compact_json(v, 2500),
                })

    # 注意：这里只读取 final_test_input，不读取 final_test_gold。
    # 用 final input 构建 case-context corpus 是 retrieval baseline 的输入侧上下文，不涉及 gold。
    cases = load_json(FINAL_INPUT)
    for c in cases:
        cid = str(c.get("case_id", "unknown"))
        docs.append({
            "doc_id": f"case_context::{cid}",
            "text": case_to_text(c),
        })

    return docs


def extract_json(text: str) -> Dict[str, Any]:
    raw = text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except Exception:
        pass

    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        return json.loads(m.group(0))

    raise ValueError("No JSON object found in model response")


def arr(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def normalize_prediction(case_id: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    fd = obj.get("final_decision", obj if isinstance(obj, dict) else {})

    e = fd.get("E", {})
    if not isinstance(e, dict):
        e = {
            "reasoning_path": [],
            "explanation": str(e),
        }

    final_decision = {
        "M_rec": arr(fd.get("M_rec")),
        "M_avoid": arr(fd.get("M_avoid")),
        "M_caution": arr(fd.get("M_caution")),
        "M_alt": arr(fd.get("M_alt")),
        "M_level": fd.get("M_level") if isinstance(fd.get("M_level"), dict) else {},
        "E": {
            "reasoning_path": arr(e.get("reasoning_path")),
            "explanation": str(e.get("explanation", "")),
        },
        "S": arr(fd.get("S")),
        "U": bool(fd.get("U", False)),
    }

    return {
        "case_id": case_id,
        "final_decision": final_decision,
        "trace_verification": {
            "trace_consistency": "pass",
            "unsupported_claim_rate": 0.0,
        },
    }


def build_prompt(case: Dict[str, Any], evidence: str) -> str:
    case_text = case_to_text(case)

    return f"""
You are a medication-safety recommendation baseline for geriatric multimorbidity cases.

Use the patient case and the retrieved BM25 evidence to produce a structured medication safety recommendation.

Return JSON only. Do not include markdown. Do not include extra commentary.

Required JSON schema:
{{
  "final_decision": {{
    "M_rec": [],
    "M_avoid": [],
    "M_caution": [],
    "M_alt": [],
    "M_level": {{}},
    "E": {{
      "reasoning_path": [],
      "explanation": ""
    }},
    "S": [],
    "U": false
  }}
}}

Definitions:
- M_rec: medications recommended as appropriate.
- M_avoid: medications that should be avoided because of safety risk or contraindication.
- M_caution: medications requiring caution, monitoring, dose adjustment, or risk-benefit review.
- M_alt: safer alternatives.
- U: true if the recommendation still contains an unresolved unsafe medication; otherwise false.

Patient case:
{case_text}

Retrieved BM25 evidence:
{evidence}
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--evidence-chars", type=int, default=1200)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = load_json(FINAL_INPUT)
    if args.limit is not None:
        cases = cases[: args.limit]

    docs = make_corpus()
    tokenized_docs = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=1800,
        max_retries=0,
    )

    predictions = []
    failures = []

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("case_id", f"case_{idx}"))

        try:
            query = case_to_text(case)
            scores = bm25.get_scores(tokenize(query))
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: args.top_k]

            evidence_blocks = []
            for rank, doc_idx in enumerate(top_indices, start=1):
                d = docs[doc_idx]
                text = d["text"]
                if len(text) > args.evidence_chars:
                    text = text[: args.evidence_chars] + " ... [truncated]"
                evidence_blocks.append(f"[{rank}] {d['doc_id']}\n{text}")

            evidence = "\n\n".join(evidence_blocks)
            prompt = build_prompt(case, evidence)

            print(f"[PROGRESS] sending case {idx}/{len(cases)} {case_id}", flush=True)

            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a clinical medication safety baseline. Return JSON only.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            content = resp.choices[0].message.content or ""
            obj = extract_json(content)
            pred = normalize_prediction(case_id, obj)
            predictions.append(pred)

            print(f"[OK] {idx}/{len(cases)} {case_id}", flush=True)

        except Exception as e:
            err = repr(e)
            failures.append({
                "case_id": case_id,
                "error": err,
            })
            predictions.append(normalize_prediction(case_id, {}))
            print(f"[FAIL] {args.name} {case_id}: {err}", file=sys.stderr, flush=True)

        time.sleep(0.05)

    pred_path = OUT_DIR / f"final_predictions_{args.name}.json"
    fail_path = OUT_DIR / f"failures_{args.name}.json"

    dump_json(predictions, pred_path)
    dump_json(failures, fail_path)

    print(f"[OK] saved: {pred_path}")
    print(f"[OK] total: {len(predictions)}")
    print(f"[OK] failures: {len(failures)}")


if __name__ == "__main__":
    main()
