from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[2]))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from baselines_v2.common import FROZEN_RULES, load_json, save_json


ROOT = Path(__file__).resolve().parents[2]
RET_DIR = ROOT / "baselines_v2" / "retrieval"
RET_DIR.mkdir(parents=True, exist_ok=True)


def build_docs(rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = []

    for i, rule in enumerate(rules.get("knowledge_rules", []) or []):
        text = json.dumps(rule, ensure_ascii=False, sort_keys=True)
        docs.append({
            "doc_id": f"knowledge_rule_{i}",
            "source": rule.get("source", rule.get("id", "knowledge_rule")),
            "kind": "knowledge_rule",
            "text": text,
            "rule": rule,
        })

    raw = rules.get("target_raw_rules", {}) or {}
    for group in ["guideline_rules", "safety_rules", "alternative_rules"]:
        for i, rule in enumerate(raw.get(group, []) or []):
            text = json.dumps(rule, ensure_ascii=False, sort_keys=True)
            docs.append({
                "doc_id": f"{group}_{i}",
                "source": rule.get("source", rule.get("rule_id", group)),
                "kind": group,
                "text": text,
                "rule": rule,
            })

    return docs


def main():
    rules = load_json(FROZEN_RULES)
    docs = build_docs(rules)

    if not docs:
        raise RuntimeError("No docs built from frozen_rules.json")

    model_path = os.environ.get("BGE_MODEL_PATH", "BAAI/bge-m3")
    print(f"[INFO] loading {model_path}")
    model = SentenceTransformer(model_path, device="cpu")

    emb = model.encode(
        [d["text"] for d in docs],
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(RET_DIR / "bge_m3_faiss.index"))
    save_json(docs, RET_DIR / "bge_m3_evidence_docs.json")

    print("[OK] docs:", len(docs))
    print("[OK] index:", RET_DIR / "bge_m3_faiss.index")
    print("[OK] docs file:", RET_DIR / "bge_m3_evidence_docs.json")


if __name__ == "__main__":
    main()
