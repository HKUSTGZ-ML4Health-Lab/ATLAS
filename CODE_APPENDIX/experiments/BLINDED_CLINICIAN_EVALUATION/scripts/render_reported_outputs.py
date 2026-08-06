#!/usr/bin/env python3
# AAAI-27 paper reference: Blinded Clinician Evaluation and Table 2.
"""Generate paper-facing CSV/Markdown/LaTeX outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CRITERIA = ["Correct.", "Safety", "Complete.", "Action.", "Evidence"]
MODELS = ["Gemini 3.1 Pro Preview", "ATLAS"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    obj = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall = obj["overall"]

    md = [
        "| Method | Correct. | Safety | Complete. | Action. | Evidence | Unsafe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        vals = [f"{float(overall[model][c]):.2f}" for c in CRITERIA]
        unsafe = overall[model]["Unsafe cases"]
        name = f"**{model}**" if model == "ATLAS" else model
        md.append(f"| {name} | " + " | ".join(vals) + f" | {unsafe['n']}/{unsafe['denominator']} |")
    (out / "reported_paper_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Blinded clinician evaluation on 40 benchmark-stratified cases. Scores are means across all reviewer--case ratings. Unsafe cases use case-level majority judgments.}",
        r"\label{tab:expert_review}",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Correct. & Safety & Complete. & Action. & Evidence & Unsafe \\",
        r"\midrule",
    ]
    for model in MODELS:
        vals = [f"{float(overall[model][c]):.2f}" for c in CRITERIA]
        unsafe = overall[model]["Unsafe cases"]
        if model == "ATLAS":
            vals = [rf"\textbf{{{v}}}" for v in vals]
            row = rf"\textbf{{ATLAS}} & " + " & ".join(vals) + rf" & \textbf{{{unsafe['n']}/{unsafe['denominator']}}} \\"
        else:
            row = model + " & " + " & ".join(vals) + f" & {unsafe['n']}/{unsafe['denominator']} " + r"\\"
        tex.append(row)
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "reported_paper_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    atlas_pref = overall["ATLAS"]["Preferred cases"]["n"]
    gem_pref = overall["Gemini 3.1 Pro Preview"]["Preferred cases"]["n"]
    ties = overall["Tie cases"]["n"]
    atlas_unsafe = overall["ATLAS"]["Unsafe cases"]["n"]
    gem_unsafe = overall["Gemini 3.1 Pro Preview"]["Unsafe cases"]["n"]
    paragraph = rf"""\subsection{{Blinded Clinician Evaluation}}

Table~\ref{{tab:expert_review}} complements the automated results with blinded
clinician judgments. ATLAS receives higher mean scores across all five
criteria. Case-level majority judgments prefer ATLAS in {atlas_pref} cases,
Gemini in {gem_pref} cases, and report {ties} ties. Reviewers flag potentially
unsafe recommendations in {atlas_unsafe} ATLAS case and {gem_unsafe} Gemini
cases. These results support clinician-rated quality within the reviewed
sample and do not establish real-world clinical effectiveness.
"""
    (out / "reported_results_paragraph.tex").write_text(paragraph, encoding="utf-8")

    print(f"[OK] Paper-facing clinician-evaluation outputs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
