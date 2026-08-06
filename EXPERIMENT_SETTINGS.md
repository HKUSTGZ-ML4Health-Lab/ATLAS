# Experiment Settings

## ATLAS

- Development cases: 39.
- Stage-1 teacher review settings: K in {1, 2, 3}, yielding 117 development
  trajectories.
- GeriMedBench patient-query budget: K=3.
- Orchestrator maximum archived review rounds: 4.
- Evidence Sufficiency Gate threshold: 0.1986607142857143, selected on Dev39
  and frozen before evaluation.
- ATLAS inference uses the frozen symbolic policy and invokes no external LLM.
- Each reported ATLAS score uses one frozen inference run per case.

## Western open-weight baseline runners

- Temperature: 0.0.
- Maximum output tokens: 2048.
- RAG top-k: 8.
- Model context length: 8192.
- Dtype: bfloat16.
- Eager execution: enabled.
- Mistral/Qwen/DeepSeek GPU memory utilization: 0.84; tensor parallelism: 1.
- MedGemma GPU memory utilization: 0.80; tensor parallelism: 2.
- Llama-3.3-70B GPU memory utilization: 0.82; tensor parallelism: 2.
- Model directory root: `ATLAS_MODEL_ROOT` or `$HOME/.cache/atlas_models`.

## GeriMedBench baseline runner defaults

- Temperature: 0.0.
- Query budget: 3.
- Maximum output tokens: 1024.
- Model context length: 4096 unless overridden.
- Dtype: bfloat16.
- Eager execution: enabled.

The executable runner scripts record the settings used for each invocation.

## Proprietary API baselines

- Models: GPT-5 (`gpt-5-2025-08-07`), Claude Opus 4.6
  (`claude-opus-4-6`), and Gemini 3.1 Pro Preview
  (`gemini-3.1-pro-preview`).
- Temperature: 0.0.
- Maximum output tokens: 2048. GeriMedBench uses 2048 for each API call.
- Top-p: not explicitly set; the provider default is used.
- Structured JSON output: enabled.
- The three models use the same prompt within each dataset.
- Parser and failure rules are shared across the three models.
- Tools, external search, and RAG are disabled.
- Inference does not access evaluation references.

### GeriMedBench API protocol

- Query budget: K=3.
- Two API calls per case and 152 calls per model.
- Call 1 selects a question or acquires information.
- Call 2 incorporates the environment answer and returns the final decision.
- Hidden facts remain inside `MedicationSafetyEnv` and are exposed only through
  `env.query()`.

