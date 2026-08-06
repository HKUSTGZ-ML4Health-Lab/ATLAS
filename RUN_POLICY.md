# Run Policy

- Dev39 is used for policy construction and calibration.
- Evaluation inputs remain separated from Dev39.
- Final inference reads the frozen input and frozen rules.
- The offline evaluator reads predictions and references after inference.
- No inference runner reads `final_test_gold.json` or another evaluation
  reference.
- Each reported score is computed from one frozen run per case.
- Every reproduction command writes its evaluation mode and generated outputs.
- Proprietary LLM comparisons use the same prompt within each dataset, the same
  structured-output parser, and the same failure rules.
- Proprietary LLM runners use no tools, external search, or RAG.

