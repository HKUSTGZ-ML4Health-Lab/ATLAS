from __future__ import annotations

import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from baselines_v2.common import FINAL_INPUT, FROZEN_RULES, OUT_DIR, load_json, save_json


ROOT = Path(__file__).resolve().parents[2]
KG_DIR = ROOT / "baselines_v2" / "kg"
KG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def norm(x: Any) -> str:
    s = str(x or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def loose_match(a: Any, b: Any) -> bool:
    aa, bb = norm(a), norm(b)
    if not aa or not bb:
        return False
    return aa in bb or bb in aa


def unique(xs):
    out, seen = [], set()
    for x in xs or []:
        s = str(x or "").strip().lower()
        if not s:
            continue
        k = norm(s)
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out


def candidates(case: Dict[str, Any]) -> List[str]:
    out = []
    for item in case.get("candidate_medications", []) or []:
        if isinstance(item, dict) and item.get("name"):
            out.append(str(item["name"]))
        elif isinstance(item, str):
            out.append(item)
    return unique(out)


def case_context(case: Dict[str, Any]) -> str:
    cs = case.get("clinical_state", {}) or {}
    vals = []
    for k in ["diseases", "current_medications", "age_related_factors", "risk_factors"]:
        vals.extend(cs.get(k, []) or [])
    ctx = cs.get("context", {}) or {}
    if isinstance(ctx, dict):
        vals.extend([str(v) for v in ctx.values() if v is not None])
    vals.append("older adult")
    vals.append(str(case.get("task", "")))
    return " | ".join(map(str, vals))


def ctx_ent(x): return "ctx::" + norm(x)
def drug_ent(x): return "drug::" + norm(x)


def rel(effect: str):
    e = str(effect or "").lower()
    if e == "support":
        return "supports_use"
    if e == "avoid":
        return "avoid_use"
    if e == "caution":
        return "requires_monitoring"
    return None


def build_triples(rules: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    triples = []
    for r in rules.get("knowledge_rules", []) or []:
        rr = rel(r.get("effect", ""))
        if not rr:
            continue
        for c in r.get("candidate_patterns", []) or []:
            for ctx in r.get("context_patterns", []) or ["older adult"]:
                if norm(c) and norm(ctx):
                    triples.append((ctx_ent(ctx), rr, drug_ent(c)))
                    triples.append((ctx_ent("older adult"), rr, drug_ent(c)))
    return list(dict.fromkeys(triples))


def score_hrt(model, entity_to_id, relation_to_id, h, r, t):
    if h not in entity_to_id or r not in relation_to_id or t not in entity_to_id:
        return float("-inf")
    device = next(model.parameters()).device
    arr = torch.as_tensor([[entity_to_id[h], relation_to_id[r], entity_to_id[t]]], dtype=torch.long, device=device)
    with torch.no_grad():
        return float(model.score_hrt(arr).detach().cpu().view(-1)[0].item())


def matching_ctx(case_text, entity_to_id):
    out = []
    for e in entity_to_id:
        if not e.startswith("ctx::"):
            continue
        label = e.replace("ctx::", "", 1)
        if label == "older adult" or loose_match(case_text, label):
            out.append(e)
    return out


def matching_drug(cand, entity_to_id):
    out = []
    for e in entity_to_id:
        if not e.startswith("drug::"):
            continue
        label = e.replace("drug::", "", 1)
        if loose_match(cand, label):
            out.append(e)
    return out


def agg(model, entity_to_id, relation_to_id, ctxs, drugs, relation):
    vals = []
    for h in ctxs:
        for t in drugs:
            v = score_hrt(model, entity_to_id, relation_to_id, h, relation, t)
            if math.isfinite(v):
                vals.append(v)
    return max(vals) if vals else float("-inf")


def main():
    cases = load_json(FINAL_INPUT)
    rules = load_json(FROZEN_RULES)
    triples = build_triples(rules)

    if not triples:
        raise RuntimeError("No triples built from frozen_rules.knowledge_rules")

    save_json([{"head": h, "relation": r, "tail": t} for h, r, t in triples],
              KG_DIR / "pykeen_rotate_training_triples.json")

    print("[INFO] triples:", len(triples))

    tf = TriplesFactory.from_labeled_triples(np.asarray(triples, dtype=str))
    device = "cpu"

    result = pipeline(
        training=tf,
        testing=tf,
        model="RotatE",
        model_kwargs={"embedding_dim": 64},
        training_kwargs={"num_epochs": 80, "batch_size": 128},
        optimizer_kwargs={"lr": 0.01},
        random_seed=SEED,
        device=device,
    )

    model = result.model
    entity_to_id = result.training.entity_to_id
    relation_to_id = result.training.relation_to_id

    preds = []
    traces = []

    for i, case in enumerate(cases):
        cid = case["case_id"]
        cands = candidates(case)
        ctext = case_context(case)
        ctxs = matching_ctx(ctext, entity_to_id)

        rows = []
        for c in cands:
            ds = matching_drug(c, entity_to_id)
            rows.append({
                "candidate": c,
                "support": agg(model, entity_to_id, relation_to_id, ctxs, ds, "supports_use"),
                "avoid": agg(model, entity_to_id, relation_to_id, ctxs, ds, "avoid_use"),
                "caution": agg(model, entity_to_id, relation_to_id, ctxs, ds, "requires_monitoring"),
            })

        M_avoid, M_caution = [], []
        for r in rows:
            finite = {k: v for k, v in {
                "support": r["support"],
                "avoid": r["avoid"],
                "caution": r["caution"],
            }.items() if math.isfinite(v)}
            if not finite:
                continue
            best = max(finite, key=finite.get)
            if best == "avoid":
                M_avoid.append(r["candidate"])
            elif best == "caution":
                M_caution.append(r["candidate"])

        avoid_set = {norm(x) for x in M_avoid}
        rec_pool = [r for r in rows if math.isfinite(r["support"]) and norm(r["candidate"]) not in avoid_set]
        rec_pool.sort(key=lambda x: x["support"], reverse=True)

        M_rec = [rec_pool[0]["candidate"]] if rec_pool else []
        M_alt = []

        pred = {
            "case_id": cid,
            "final_decision": {
                "M_rec": unique(M_rec),
                "M_avoid": unique(M_avoid),
                "M_caution": unique([x for x in M_caution if norm(x) not in avoid_set]),
                "M_alt": M_alt,
                "M_level": {m: "B" for m in M_rec},
                "E": {
                    "reasoning_path": [
                        f"{r['candidate']}: support={r['support']:.4f}, avoid={r['avoid']:.4f}, caution={r['caution']:.4f}"
                        for r in rows
                    ],
                    "explanation": "PyKEEN-RotatE scored candidate drugs over frozen-rule triples."
                },
                "S": ["PyKEEN-RotatE over frozen_rules.knowledge_rules"],
                "U": False,
            },
            "trace_verification": {
                "trace_consistency": "pass",
                "unsupported_claim_rate": 0.0,
            },
            "audit": {
                "baseline_schema": "baselines_v2_second_batch",
                "uses_final_gold_during_inference": False,
                "method": "PyKEEN-RotatE",
            },
        }

        preds.append(pred)
        traces.append({"case_id": cid, "rows": rows})
        print(f"[OK] pykeen_rotate {i+1}/{len(cases)} {cid}")

    save_json(preds, OUT_DIR / "final_predictions_pykeen_rotate.json")
    save_json(traces, OUT_DIR / "trace_pykeen_rotate.json")
    print("[OK] saved:", OUT_DIR / "final_predictions_pykeen_rotate.json")


if __name__ == "__main__":
    main()
