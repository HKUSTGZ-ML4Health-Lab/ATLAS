# Core Safety-Reasoning Ablations

| Variant | N | Strict Success | M_rec F1 | M_avoid Recall | M_avoid F1 | M_caution F1 | M_alt F1 | Unsafe Rate | Safety | Trace Pass Rate | OSRS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full ATLAS | 201 | 92.04 | 92.50 | 100.00 | 91.57 | 75.47 | 92.84 | 0.00 | 100.00 | 92.04 | 97.21 |
| w/o PMCG | 201 | 19.90 | 48.26 | 48.26 | 48.26 | 40.30 | 48.26 | 51.74 | 48.26 | 100.00 | 48.78 |
| w/o Geriatric Risk Auditor | 201 | 38.81 | 92.70 | 93.53 | 87.40 | 40.30 | 92.70 | 6.47 | 93.53 | 100.00 | 80.47 |
| w/o Drug Conflict Auditor | 201 | 69.15 | 49.59 | 69.15 | 69.15 | 59.70 | 49.59 | 30.85 | 69.15 | 100.00 | 74.41 |
| w/o Safety Gate | 201 | 26.87 | 73.13 | 73.13 | 73.13 | 40.30 | 73.13 | 26.87 | 73.13 | 98.01 | 65.90 |

Official paper-facing rows: Full ATLAS, w/o PMCG, w/o Geriatric Risk Auditor, w/o Drug Conflict Auditor, and w/o Safety Gate.
