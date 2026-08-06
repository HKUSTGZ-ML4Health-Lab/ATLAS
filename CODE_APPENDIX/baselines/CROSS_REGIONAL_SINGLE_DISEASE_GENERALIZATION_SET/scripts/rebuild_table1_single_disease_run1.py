#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "baselines_12" / "results"

ORDER = [
    ("Non-neural retrieval", "Retrieval-only Evidence Engine", "summary_retrieval_only_engine.json"),
    ("Symbolic system", "Frozen Guideline Engine", "summary_frozen_guideline.json"),
    ("General open LLM", "Mistral-Small-3.2-24B-Instruct-2506", "summary_mistral_small_32_24b_llm_only.json"),
    ("Reasoning open LLM", "Qwen3-30B-A3B-Instruct-2507", "summary_qwen3_30b_a3b_instruct_2507_llm_only.json"),
    ("Reasoning open LLM", "DeepSeek-R1-Distill-Qwen-32B", "summary_deepseek_r1_distill_qwen32b_llm_only.json"),
    ("Medical text LLM", "MedGemma 27B Text", "summary_medgemma27b_text_llm_only.json"),
    ("Retrieval-based", "BM25 + Llama-3.3-70B", "summary_bm25_llama33_70b_rag.json"),
    ("Dense RAG", "BGE-M3 + FAISS + MedGemma 27B", "summary_bge_m3_faiss_medgemma27b_rag.json"),
    ("Graph-based", "Generic KG Recommendation", "summary_generic_kg.json"),
    ("KGE-based", "PyKEEN-RotatE", "summary_pykeen_rotate.json"),
    ("Medical multi-agent", "MDAgents + Shared RAG", "summary_mdagents_shared_rag.json"),
    ("Ours", "ATLAS", "summary_atlas.json"),
]
COLS = ["Category","Method","Status","N","Strict Success ↑","M_rec F1 ↑","M_avoid Recall ↑","M_avoid F1 ↑","M_caution F1 ↑","M_alt F1 ↑","Unsafe Rate ↓","Trace Pass ↑","OSRS ↑"]
rows = []
for cat, method, fn in ORDER:
    s = json.load(open(R / fn, encoding="utf-8"))
    rows.append({
        "Category": cat, "Method": method, "Status": "OK", "N": int(s["N"]),
        "Strict Success ↑": float(s["success_rate_strict"]),
        "M_rec F1 ↑": float(s["M_rec_f1"]),
        "M_avoid Recall ↑": float(s["M_avoid_recall"]),
        "M_avoid F1 ↑": float(s["M_avoid_f1"]),
        "M_caution F1 ↑": float(s["M_caution_f1"]),
        "M_alt F1 ↑": float(s["M_alt_f1"]),
        "Unsafe Rate ↓": float(s["unsafe_rate"]),
        "Trace Pass ↑": float(s["trace_pass_rate"]),
        "OSRS ↑": float(s["OSRS"]),
    })
with open(R / "table1_single_disease_run1.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
with open(R / "table1_single_disease_run1.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader(); w.writerows(rows)
def fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)
lines = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"]+["---:"]*(len(COLS)-1)) + "|"]
for r in rows:
    lines.append("| " + " | ".join(fmt(r[c]) for c in COLS) + " |")
(R / "table1_single_disease_run1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("[OK] regenerated", R / "table1_single_disease_run1.md")
print((R / "table1_single_disease_run1.md").read_text(encoding="utf-8"))
