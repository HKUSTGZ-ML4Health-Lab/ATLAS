# Metrics

## Western Multimorbidity Evaluation Set, N=201

The main static table reports Strict Success, micro-aggregated M_rec F1,
M_avoid Recall, M_avoid F1, M_caution F1, and M_alt F1, plus case-level Unsafe
Recommendation Rate, Trace Pass Rate, and OSRS.

Strict Success requires recommendation, avoidance, caution, alternative,
unsafe, and trace criteria to pass jointly for a case.

## Cross-Regional Single-Disease Generalization Set, N=612

The formal single-disease result reports **macro-aggregated** component metrics:
M_rec F1, M_avoid Recall, M_avoid F1, M_caution F1, and M_alt F1. Strict Success,
Unsafe Rate, Trace Pass Rate, and OSRS remain case-level or aggregate safety
measures. The evaluator may also emit compact micro diagnostics; those are not
the component values reported as the formal N=612 result.

## OSRS

OSRS is reported on a 0--100 scale:

`0.25 R_avoid + 0.20 R_rec + 0.15 R_caution + 0.15 R_alt + 0.15 T + 0.10 (1-U)`

Avoidance recall receives the largest weight because a false negative omits a
medication that the patient should avoid. F1 is used for the other decision
components to penalize both omissions and unsupported additions.

## GeriMedBench

GeriMedBench reports Revision Accuracy, Final Strict, Unsafe Rate, Trace
Consistency, and Agent OSRS. Information Gain and Query Efficiency are retained
as supplementary acquisition metrics.
