# Reviewer-Level Inputs

For reviewer-level aggregation, place the de-identified files here:

```text
private_inputs/
├── PRIVATE_randomization_key.csv
└── ratings/
    ├── reviewer_R1_ratings.csv
    ├── reviewer_R2_ratings.csv
    └── reviewer_R3_ratings.csv
```

Then run:

```bash
bash CODE_APPENDIX/reproduce_BLINDED_CLINICIAN_EVALUATION.sh
```

