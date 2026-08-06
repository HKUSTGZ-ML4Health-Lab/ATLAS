# Determinism and Seeds

The standalone ATLAS inference path is symbolic and deterministic. It invokes
no sampling-based external model. Reproduction wrappers set:

```bash
PYTHONHASHSEED=0
```

The blinded clinician aggregation script exposes an explicit seed for bootstrap
resampling when de-identified reviewer-level inputs are supplied. Baseline LLM
runners use temperature 0.0 and expose their runtime controls through command
line arguments and environment variables.
