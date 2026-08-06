# ATLAS Method Overview

ATLAS is a coupled graph-policy framework for medication safety in older adults
with multimorbidity. The implementation follows four stages.

## Stage I: Clinical intake and treatment goal

The Clinical State Grounder converts the visible case into a structured patient
state containing conditions, current medications, age and geriatric factors,
safety modifiers, and therapeutic context. Unreported information remains
unknown.

## Stage II: Patient-specific medication conflict graph

The PMCG Builder selects guideline relations relevant to the current patient
state and marks decision-changing unresolved relations. In the interactive
setting, these relations determine the next targeted question. New evidence
updates the patient state and the graph.

## Stage III: Risk-first medication decision

The Drug Conflict Auditor screens medication-condition conflicts. The Geriatric
Risk Auditor assesses age-related cautions and monitoring needs. Avoidance has
priority over caution and recommendation. The Alternative Agent searches for a
safer option when a candidate is excluded.

## Stage IV: Reconciliation and verification

The Revision Agent resolves component overlap, the Trace Verifier checks the
evidence path, and the Safety Gate applies the final consistency checks.

## Pseudocode

```text
input: visible patient state s, treatment goal g, guideline graph G
P <- personalize(G, s, g)
while an unresolved relation can change the decision and query budget remains:
    q <- highest-priority informative question(P)
    a <- environment_answer(q)
    s <- update_patient_state(s, q, a)
    P <- personalize(G, s, g)

avoid   <- DrugConflictAuditor(P, s)
caution <- GeriatricRiskAuditor(P, s)
recommend <- reconcile_candidates(P, avoid, caution)
alternative <- AlternativeAgent(P, avoid)
plan <- RevisionAgent(recommend, avoid, caution, alternative)
trace <- TraceVerifier(plan, P)
return SafetyGate(plan, trace)
```

The paper-visible components and their source files are listed in
`PAPER_COMPONENT_MAPPING.md`.
