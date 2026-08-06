from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "baselines_v2" / "results"

OUT_CSV = RESULT_DIR / "table2_ablations.csv"
OUT_MD = RESULT_DIR / "table2_ablations.md"
OUT_JSON = RESULT_DIR / "table2_ablations.json"


ROWS = [
    ("Full ATLAS", "atlas"),
    ("w/o PMCG", "ablation_no_pmcg"),
    ("w/o Drug Conflict Auditor", "ablation_no_drug_conflict_auditor"),
    ("w/o Geriatric Risk Auditor", "ablation_no_geriatric_risk_auditor"),
    ("w/o Safety Gate", "ablation_no_safety_gate"),
    ("w/o Alternative Agent", "ablation_no_alternative_agent"),
    ("w/o Safety Critic + Revision", "ablation_no_safety_critic_revision"),
    ("w/o Open Candidate Generation", "ablation_no_open_candidate_generation"),
    ("w/o TraceVerifier", "ablation_no_trace_verifier"),
]


HEADERS = [
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
    "ΔOSRS",
    "ΔStrict",
    "ΔUnsafe",
]


KEYS = {
    "N": "N",
    "Strict Success ↑": "success_rate_strict",
    "M_rec F1 ↑": "M_rec_f1",
    "M_avoid Recall ↑": "M_avoid_recall",
    "M_avoid F1 ↑": "M_avoid_f1",
    "M_caution F1 ↑": "M_caution_f1",
    "M_alt F1 ↑": "M_alt_f1",
    "Unsafe Rate ↓": "unsafe_rate",
    "Trace Pass ↑": "trace_pass_rate",
    "OSRS ↑": "OSRS",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def val(summary, key):
    x = summary.get(key)
    if x is None:
        return "MISSING"
    if isinstance(x, int):
        return x
    return round(float(x), 2)


def row_from_summary(method, name, full=None):
    p = RESULT_DIR / f"summary_{name}.json"

    if not p.exists():
        row = {h: "MISSING" for h in HEADERS}
        row["Method"] = method
        row["Status"] = f"MISSING: {p.name}"
        return row

    s = load_json(p)

    row = {
        "Method": method,
        "Status": "OK",
    }

    for out_key, json_key in KEYS.items():
        row[out_key] = val(s, json_key)

    if row["N"] != 201:
        row["Status"] = f"CHECK_N={row['N']}"

    if full is None:
        row["ΔOSRS"] = 0.0
        row["ΔStrict"] = 0.0
        row["ΔUnsafe"] = 0.0
    else:
        row["ΔOSRS"] = round(float(row["OSRS ↑"]) - float(full["OSRS ↑"]), 2)
        row["ΔStrict"] = round(float(row["Strict Success ↑"]) - float(full["Strict Success ↑"]), 2)
        row["ΔUnsafe"] = round(float(row["Unsafe Rate ↓"]) - float(full["Unsafe Rate ↓"]), 2)

    return row


def write_md(rows, path):
    lines = []
    lines.append("| " + " | ".join(HEADERS) + " |")
    lines.append("| " + " | ".join(["---"] * len(HEADERS)) + " |")

    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in HEADERS) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    full = row_from_summary("Full ATLAS", "atlas", full=None)

    rows = [full]
    for method, name in ROWS[1:]:
        rows.append(row_from_summary(method, name, full=full))

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(rows, OUT_MD)

    print(OUT_MD.read_text(encoding="utf-8"))
    print(f"[OK] saved {OUT_CSV}")
    print(f"[OK] saved {OUT_MD}")
    print(f"[OK] saved {OUT_JSON}")

    bad = [r for r in rows if r["Status"] != "OK"]
    if bad:
        raise SystemExit("[ERROR] Some ablation rows are not OK.")


if __name__ == "__main__":
    main()
