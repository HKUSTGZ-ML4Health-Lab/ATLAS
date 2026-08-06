# Proprietary Large-Language-Model Baselines

## Model snapshots

| Model | Exact API model ID | Provider |
|---|---|---|
| GPT-5 | `gpt-5-2025-08-07` | OpenAI |
| Claude Opus 4.6 | `claude-opus-4-6` | Anthropic |
| Gemini 3.1 Pro Preview | `gemini-3.1-pro-preview` | Google |

## Shared evaluation settings

| Setting | Western N=201 | GeriMedBench N=76 | Single-Disease N=612 |
|---|---:|---:|---:|
| Temperature | 0.0 | 0.0 | 0.0 |
| Maximum output tokens | 2048 | 2048 per API call | 2048 |
| Top-p | Provider default; not explicitly set | Provider default; not explicitly set | Provider default; not explicitly set |
| Structured JSON output | Yes | Yes | Yes |
| Prompt parity | Same prompt for the three models within each dataset | Same prompt for the three models within each dataset | Same prompt for the three models within each dataset |
| Shared parser | Yes | Yes | Yes |
| Shared failure rules | Yes | Yes | Yes |
| Tools, search, or RAG | None | None | None |
| Evaluation-reference access during inference | None | None | None |

## GeriMedBench interaction protocol

- Query budget: `K=3`.
- One complete trajectory is retained per case.
- Two API calls are used per case.
- The first call performs question selection or information acquisition.
- The second call uses the environment response to update the state and return
  the final structured decision.
- Total calls per model: `76 x 2 = 152`.
- Hidden safety facts are accessible only to `MedicationSafetyEnv`.
- Models obtain hidden facts only through `env.query()`.
- Gold/reference records are read only by the locked offline evaluator after
  inference.

## Completion status

| Dataset | GPT-5 | Claude Opus 4.6 | Gemini 3.1 Pro Preview |
|---|---|---|---|
| Western Multimorbidity Evaluation Set | 201/201, complete | 201/201, complete | 201/201, complete |
| GeriMedBench | 76/76, complete | 76/76, complete | 76/76, complete |
| Cross-Regional Single-Disease Generalization Set | 612/612, complete | 612/612, complete | 612/612, complete |

## Western Multimorbidity Evaluation Set

| Method | N | Strict Success | M_rec F1 | M_avoid Recall | M_avoid F1 | M_caution F1 | M_alt F1 | Unsafe Rate | Trace | OSRS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 | 201 | 32.34 | 69.15 | 99.00 | 90.05 | 41.29 | 49.12 | 0.50 | 100.00 | 71.37 |
| Claude Opus 4.6 | 201 | 34.83 | 88.56 | 98.01 | 97.35 | 46.77 | 85.32 | 0.00 | 100.00 | 82.30 |
| Gemini 3.1 Pro Preview | 201 | 38.31 | 63.68 | 99.00 | 97.84 | 70.65 | 80.60 | 0.00 | 100.00 | 82.58 |

## Cross-Regional Single-Disease Generalization Set

| Method | Completed | Strict Success | M_rec F1 | M_avoid Recall | M_avoid F1 | M_caution F1 | M_alt F1 | Unsafe Rate | Trace | OSRS | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| GPT-5 | 612/612 | 76.47 | 94.28 | 98.86 | 98.86 | 92.48 | 82.19 | 0.00 | 100.00 | 94.77 | Complete |
| Claude Opus 4.6 | 612/612 | 81.21 | 98.37 | 98.20 | 98.15 | 88.73 | 84.31 | 0.00 | 100.00 | 95.18 | Complete |
| Gemini 3.1 Pro Preview | 612/612 | 92.60 | 97.19 | 98.98 | 98.98 | 95.41 | 95.66 | 0.00 | 100.00 | 97.84 | Complete |

## GeriMedBench

| Method | N | Information Gain | Query Efficiency | Revision Accuracy | Final Strict | Unsafe Rate | Trace Consistency | Agent OSRS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-5 | 76 | 71.05 | 33.33 | 40.60 | 2.63 | 0.00 | 81.58 | 58.98 |
| Claude Opus 4.6 | 76 | 80.92 | 38.60 | 40.51 | 5.26 | 0.00 | 81.58 | 62.87 |
| Gemini 3.1 Pro Preview | 76 | 74.34 | 35.53 | 43.18 | 7.89 | 2.63 | 73.68 | 60.54 |

