# Datasets and Protocols

- Development Set: 39 cases, used for policy construction and calibration.
- Western Multimorbidity Evaluation Set: N=201, non-interactive.
- GeriMedBench / Asian Multimorbidity Evaluation Set: N=76, interactive, K=3.
- Cross-Regional Single-Disease Generalization Set: N=612, non-interactive,
  candidate-constrained cross-guideline evaluation.

Inference does not read evaluation references. Gold/reference data first enter
the workflow in the frozen offline evaluator. The complete archived records are
listed in `DATA_APPENDIX/DATA_APPENDIX_MANIFEST.json`.
