# Paper-to-Code Component Mapping

| Paper component | Implementation |
|---|---|
| Unified Orchestrator | `CODE_APPENDIX/framework_src/agents/unified_orchestrator.py` |
| Clinical State Grounder | `CODE_APPENDIX/framework_src/agents/clinical_state_grounder.py`; `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| Drug Conflict Auditor | `CODE_APPENDIX/framework_src/agents/drug_conflict_auditor.py`; `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| Geriatric Risk Auditor | `CODE_APPENDIX/framework_src/agents/geriatric_risk_auditor.py`; `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| Alternative Agent | `CODE_APPENDIX/framework_src/agents/alternative_agent.py`; `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| Revision Agent | `CODE_APPENDIX/framework_src/agents/revision_agent.py`; `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| Trace Verifier | `CODE_APPENDIX/framework_src/agents/trace_verifier.py`; `CODE_APPENDIX/framework_src/system_impl/agents_impl.py` |
| Safety Gate | `safety_critic.py`, `evidence_sufficiency_gate.py`, `revision_agent.py`, `trace_verifier.py`, and the orchestrator final checks |
| PMCG Builder | `CODE_APPENDIX/framework_src/pmcg/builder.py` |

Core method files include paper-section mapping headers. The policy-distillation pipeline is located under
`CODE_APPENDIX/01_DEV39_WORKSPACE/policy_distillation/`.
