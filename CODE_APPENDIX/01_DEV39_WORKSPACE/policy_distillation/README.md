# Policy Distillation and Freeze Verification

This directory reproduces the teacher trajectories and runs
exact policy-freeze verification for the released runtime artifacts.

- Development cases: 39.
- Teacher review settings: K in {1, 2, 3}.
- Total trajectories: 117.
- Evaluation-reference access during policy construction: none.
- Frozen runtime policy: `02_FROZEN_INFERENCE/frozen/frozen_rules.json`.

The verifier checks complete case-budget coverage, confirms that the development
and frozen runtime policies are exact matches, confirms that the reported
inference entry point loads the frozen policy, and verifies that released
experiment artifacts remain unchanged.

Run:

```bash
bash CODE_APPENDIX/reproduce_POLICY_DISTILLATION_STAGE1.sh
```

Verified policy SHA-256:

```text
24ad9d60e54ac8c1c58fbbc91727c66c991c1ea9febc97da698a6e41b06c603a
```
