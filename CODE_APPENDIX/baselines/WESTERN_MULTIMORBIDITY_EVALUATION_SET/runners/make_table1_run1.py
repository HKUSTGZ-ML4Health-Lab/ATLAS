from __future__ import annotations

import csv
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("ATLAS_ROOT", str(Path(__file__).resolve().parents[3]))).expanduser().resolve()
RESULT_DIR = ROOT / "baselines_v2/results"

OUT_CSV = RESULT_DIR / "table1_run1.csv"
OUT_MD = RESULT_DIR / "table1_run1.md"
OUT_JSON = RESULT_DIR / "table1_run1.json"


ROWS = [
    {"Category": "Non-neural retrieval", "Method": "Retrieval-only Evidence Engine", "names": ["retrieval_only_engine"]},
    {"Category": "Symbolic system", "Method": "Frozen Guideline Engine", "names": ["frozen_guideline"]},
    {"Category": "General open LLM", "Method": "Mistral-Small-3.2-24B-Instruct-2506", "names": ["mistral_small_32_24b_llm_only"]},
    {"Category": "Reasoning open LLM", "Method": "Qwen3-30B-A3B-Instruct-2507", "names": ["qwen3_30b_a3b_instruct_2507_llm_only"]},
    {"Category": "Reasoning open LLM", "Method": "DeepSeek-R1-Distill-Qwen-32B", "names": ["deepseek_r1_distill_qwen32b_llm_only"]},
    {"Category": "Medical text LLM", "Method": "MedGemma 27B Text", "names": ["medgemma27b_text_llm_only"]},
    {"Category": "Retrieval-based", "Method": "BM25 + Llama-3.3-70B", "names": ["bm25_llama33_70b_rag"]},
    {"Category": "Dense RAG", "Method": "BGE-M3 + FAISS + MedGemma 27B", "names": ["bge_m3_faiss_medgemma27b_rag"]},
    {"Category": "Graph-based", "Method": "Generic KG Recommendation", "names": ["generic_kg"]},
    {"Category": "KGE-based", "Method": "PyKEEN-RotatE", "names": ["pykeen_rotate"]},
    {"Category": "Medical multi-agent", "Method": "MDAgents + Shared RAG", "names": ["mdagents_shared_rag"]},
    {"Category": "Ours", "Method": "ATLAS", "names": ["atlas"]},
]


HEADERS = [
    "Category",
    "Method",
    "Status",
    "N",
    "Strict Success ↑",
    "M_rec F1 ↑",
    "M_avoid Recall ↑",
    "M_avoid F1 ↑",
    "M_caution F1 ↑",
    "M_alt F1 ↑",
    "Unsafe Rate ↓",
    "Trace Pass ↑",
    "OSRS ↑",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct_from_summary(x):
    """
    summary_*.json from evaluate_one.py already stores percentage values.
    Do not multiply 0-1 values again.
    Example: 1.0 means 1.0%, not 100%.
    """
    if x is None:
        return "MISSING"
    return round(float(x), 2)


def pct_from_eval(x):
    """
    eval_*.json may store micro metrics as fractions.
    Convert only eval fraction-style values.
    """
    if x is None:
        return "MISSING"
    x = float(x)
    if 0 <= x <= 1:
        x *= 100
    return round(x, 2)


def num(x):
    if x is None:
        return "MISSING"
    return int(x)


def find_result(names):
    for name in names:
        p = RESULT_DIR / f"summary_{name}.json"
        if p.exists():
            return p, "summary"

    for name in names:
        p = RESULT_DIR / f"eval_{name}.json"
        if p.exists():
            return p, "eval"

    return None, None


def compact_to_row(category, method, data):
    return {
        "Category": category,
        "Method": method,
        "Status": "OK",
        "N": num(data.get("total_cases", data.get("N"))),
        "Strict Success ↑": pct_from_summary(data.get("success_rate_strict")),
        "M_rec F1 ↑": pct_from_summary(data.get("M_rec_f1")),
        "M_avoid Recall ↑": pct_from_summary(data.get("M_avoid_recall")),
        "M_avoid F1 ↑": pct_from_summary(data.get("M_avoid_f1")),
        "M_caution F1 ↑": pct_from_summary(data.get("M_caution_f1")),
        "M_alt F1 ↑": pct_from_summary(data.get("M_alt_f1")),
        "Unsafe Rate ↓": pct_from_summary(data.get("unsafe_rate", data.get("unsafe_recommendation_rate"))),
        "Trace Pass ↑": pct_from_summary(data.get("trace_pass_rate", data.get("trace_consistency_pass_rate"))),
        "OSRS ↑": pct_from_summary(data.get("OSRS")),
    }


def eval_to_row(category, method, data):
    s = data["summary"]
    micro = s["micro"]

    return {
        "Category": category,
        "Method": method,
        "Status": "OK",
        "N": num(s.get("total_cases")),
        "Strict Success ↑": pct_from_eval(s.get("success_rate_strict")),
        "M_rec F1 ↑": pct_from_eval(micro["M_rec"]["f1"]),
        "M_avoid Recall ↑": pct_from_eval(micro["M_avoid"]["recall"]),
        "M_avoid F1 ↑": pct_from_eval(micro["M_avoid"]["f1"]),
        "M_caution F1 ↑": pct_from_eval(micro["M_caution"]["f1"]),
        "M_alt F1 ↑": pct_from_eval(micro["M_alt"]["f1"]),
        "Unsafe Rate ↓": pct_from_eval(s.get("unsafe_recommendation_rate")),
        "Trace Pass ↑": pct_from_eval(s.get("trace_consistency_pass_rate")),
        "OSRS ↑": pct_from_eval(s["overall_safety_reasoning"]["overall_safety_reasoning_score_percent"]),
    }


def missing_row(category, method, names):
    return {
        "Category": category,
        "Method": method,
        "Status": "MISSING_RESULT: " + ",".join(names),
        "N": "MISSING",
        "Strict Success ↑": "MISSING",
        "M_rec F1 ↑": "MISSING",
        "M_avoid Recall ↑": "MISSING",
        "M_avoid F1 ↑": "MISSING",
        "M_caution F1 ↑": "MISSING",
        "M_alt F1 ↑": "MISSING",
        "Unsafe Rate ↓": "MISSING",
        "Trace Pass ↑": "MISSING",
        "OSRS ↑": "MISSING",
    }


def write_md(rows, path):
    lines = []
    lines.append("| " + " | ".join(HEADERS) + " |")
    lines.append("| " + " | ".join(["---"] * len(HEADERS)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in HEADERS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for item in ROWS:
        path, kind = find_result(item["names"])

        if path is None:
            rows.append(missing_row(item["Category"], item["Method"], item["names"]))
            continue

        data = load_json(path)

        if kind == "eval" or ("summary" in data and "micro" in data.get("summary", {})):
            row = eval_to_row(item["Category"], item["Method"], data)
        else:
            row = compact_to_row(item["Category"], item["Method"], data)

        if row["N"] != 201:
            row["Status"] = f"CHECK_N={row['N']}"

        rows.append(row)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    write_md(rows, OUT_MD)

    print(OUT_MD.read_text(encoding="utf-8"))
    print(f"[OK] CSV saved to: {OUT_CSV}")
    print(f"[OK] MD saved to: {OUT_MD}")
    print(f"[OK] JSON saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
