# ATLAS Code Appendix

Run all paper-aligned experiment checks from this directory:

```bash
bash reproduce_all.sh
```

The wrapper creates required output directories and runs:

1. Dev39 teacher-trajectory reproduction and exact policy-freeze verification.
2. Western N=201 deterministic inference and offline evaluation.
3. GeriMedBench N=76.
4. Single-disease N=612.
5. Core Safety-Reasoning Ablations.
6. Blinded clinician evaluation, N=40.

Generated outputs are written to `reproduced/`. Baseline source runners are
available under `baselines/`; model paths and service endpoints are configured
through their command-line arguments and environment variables.
