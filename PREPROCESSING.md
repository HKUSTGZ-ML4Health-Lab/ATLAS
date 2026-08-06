# Preprocessing and Canonicalization

The released pipeline includes the preprocessing required by the ATLAS
experiments.

| Function | Primary implementation |
|---|---|
| Patient-state parsing | `CODE_APPENDIX/framework_src/agents/clinical_state_grounder.py` |
| Alias and entity normalization | `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| Negation and uncertainty handling | `CODE_APPENDIX/framework_src/agents/clinical_state_grounder.py` |
| Candidate normalization | `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| PMCG construction | `CODE_APPENDIX/framework_src/pmcg/builder.py` |
| Structured-output parsing | baseline runners under `CODE_APPENDIX/baselines/` |
| Offline metric preparation | evaluator scripts under `CODE_APPENDIX/03_OFFLINE_EVALUATION/` and `CODE_APPENDIX/experiments/` |

Development preprocessing reads only the 39-case development workspace. Final
inference reads the frozen input and frozen rules. Test references are consumed
only by the offline evaluation stage.
