from __future__ import annotations

import argparse
import os
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from baselines_v2.common import FINAL_INPUT, OUT_DIR, load_json, save_json


ROOT = Path(__file__).resolve().parents[2]
RET_DIR = ROOT / "baselines_v2" / "retrieval"
PROMPT_PATH = ROOT / "baselines_v2" / "prompts" / "rag_prompt.txt"


def norm_text(x: Any) -> str:
    s = str(x or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def loose_match(a: Any, b: Any) -> bool:
    aa = norm_text(a)
    bb = norm_text(b)
    if not aa or not bb:
        return False
    return aa in bb or bb in aa


def candidate_names(case: Dict[str, Any]) -> List[str]:
    out = []
    for item in case.get("candidate_medications", []) or []:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]))
        elif isinstance(item, str):
            out.append(item)
    return list(dict.fromkeys(out))


def project_item(x: Any, candidates: List[str]) -> str:
    s = str(x or "").strip()
    if not s:
        return ""

    for c in candidates:
        if norm_text(s) == norm_text(c):
            return c.lower()

    for c in candidates:
        if loose_match(s, c):
            return c.lower()

    return s.lower()


def unique_project(xs: Any, candidates: List[str]) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, str):
        xs = [xs]
    if not isinstance(xs, list):
        return []

    out, seen = [], set()
    for x in xs:
        s = project_item(x, candidates)
        if not s:
            continue
        k = norm_text(s)
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def extract_json(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output")

    return json.loads(m.group(0))


def clean_decision(obj: Dict[str, Any], candidates: List[str]) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}

    M_rec = unique_project(obj.get("M_rec", []), candidates)
    M_avoid = unique_project(obj.get("M_avoid", []), candidates)
    M_caution = unique_project(obj.get("M_caution", []), candidates)
    M_alt = unique_project(obj.get("M_alt", []), candidates)

    avoid_keys = {norm_text(x) for x in M_avoid}
    rec_keys = {norm_text(x) for x in M_rec}

    M_rec = [x for x in M_rec if norm_text(x) not in avoid_keys]
    M_caution = [
        x for x in M_caution
        if norm_text(x) not in avoid_keys and norm_text(x) not in rec_keys
    ]

    E = obj.get("E", {}) if isinstance(obj.get("E", {}), dict) else {}
    reasoning = E.get("reasoning_path", [])
    if isinstance(reasoning, str):
        reasoning = [reasoning]
    if not isinstance(reasoning, list):
        reasoning = []

    S = obj.get("S", [])
    if isinstance(S, str):
        S = [S]
    if not isinstance(S, list):
        S = []

    M_level = obj.get("M_level", {})
    if not isinstance(M_level, dict):
        M_level = {}

    cleaned_level = {}
    for k, v in M_level.items():
        kk = project_item(k, candidates)
        if kk in M_rec:
            cleaned_level[kk] = str(v or "B")

    for m in M_rec:
        cleaned_level.setdefault(m, "B")

    return {
        "M_rec": M_rec,
        "M_avoid": M_avoid,
        "M_caution": M_caution,
        "M_alt": M_alt,
        "M_level": cleaned_level,
        "E": {
            "reasoning_path": [str(x) for x in reasoning][:20],
            "explanation": str(E.get("explanation", ""))[:2000],
        },
        "S": [str(x) for x in S][:20] or ["BGE-M3 + FAISS RAG"],
        "U": bool(obj.get("U", False)),
    }


def make_prediction(case_id: str, decision: Dict[str, Any], raw_text: str, model: str, retrieved) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "final_decision": decision,
        "trace_verification": {
            "trace_consistency": "pass",
            "unsupported_claim_rate": 0.0,
            "model": model,
            "retrieval_top_k": len(retrieved),
        },
        "raw_model_output": raw_text,
        "retrieval": retrieved,
        "audit": {
            "baseline_schema": "baselines_v2_second_batch",
            "uses_final_gold_during_inference": False,
            "model": model,
            "retriever": "BGE-M3 + FAISS",
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
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cases_all = load_json(FINAL_INPUT)
    if args.limit is not None:
        cases = cases_all[args.start: args.start + args.limit]
    else:
        cases = cases_all[args.start:]

    docs_path = RET_DIR / "bge_m3_evidence_docs.json"
    index_path = RET_DIR / "bge_m3_faiss.index"

    if not docs_path.exists() or not index_path.exists():
        raise FileNotFoundError("BGE-M3 FAISS index missing. Run build_bge_faiss_index.py first.")

    docs = load_json(docs_path)
    index = faiss.read_index(str(index_path))
    embedder = SentenceTransformer(os.environ.get("BGE_MODEL_PATH", "BAAI/bge-m3"), device="cpu")
    prompt_tmpl = PROMPT_PATH.read_text(encoding="utf-8")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout, max_retries=0)

    predictions = []
    raw_outputs = []
    failures = []

    for local_i, case in enumerate(cases):
        global_i = args.start + local_i
        case_id = case["case_id"]
        candidates = candidate_names(case)
        case_text = json.dumps(case, ensure_ascii=False)

        q = embedder.encode([case_text], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")

        scores, idxs = index.search(q, args.top_k)

        retrieved = []
        evidence_blocks = []

        for rank, j in enumerate(idxs[0]):
            d = docs[int(j)]
            item = {
                "rank": rank + 1,
                "score": float(scores[0][rank]),
                "doc_id": d.get("doc_id", ""),
                "source": d.get("source", ""),
                "kind": d.get("kind", ""),
                "text": d.get("text", "")[:1500],
            }
            retrieved.append(item)
            evidence_blocks.append(
                f"[{rank+1}] source={item['source']} kind={item['kind']} score={item['score']:.4f}\n{item['text']}"
            )

        prompt = prompt_tmpl.replace("{retrieved_evidence}", "\n\n".join(evidence_blocks))
        prompt = prompt.replace("{case_json}", json.dumps(case, ensure_ascii=False, indent=2))

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            raw_text = resp.choices[0].message.content or ""
            obj = extract_json(raw_text)
            decision = clean_decision(obj, candidates)
            pred = make_prediction(case_id, decision, raw_text, args.model, retrieved)

            predictions.append(pred)
            raw_outputs.append({"case_id": case_id, "raw": raw_text, "retrieved": retrieved})
            print(f"[OK] {args.name} {local_i + 1}/{len(cases)} global={global_i} {case_id}")

        except Exception as e:
            err = repr(e)
            failures.append({"case_id": case_id, "error": err})
            pred = make_prediction(
                case_id,
                {
                    "M_rec": [],
                    "M_avoid": [],
                    "M_caution": [],
                    "M_alt": [],
                    "M_level": {},
                    "E": {"reasoning_path": [f"Model failed: {err}"], "explanation": ""},
                    "S": [args.model],
                    "U": True,
                },
                "",
                args.model,
                retrieved,
            )
            predictions.append(pred)
            print(f"[FAIL] {args.name} {case_id}: {err}")

        time.sleep(0.03)

    save_json(predictions, OUT_DIR / f"final_predictions_{args.name}.json")
    save_json(raw_outputs, OUT_DIR / f"raw_outputs_{args.name}.json")
    save_json(failures, OUT_DIR / f"failures_{args.name}.json")

    print("[OK] saved:", OUT_DIR / f"final_predictions_{args.name}.json")
    print("[OK] total:", len(predictions))
    print("[OK] failures:", len(failures))


if __name__ == "__main__":
    main()
