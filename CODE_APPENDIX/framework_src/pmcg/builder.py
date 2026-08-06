# AAAI-27 paper reference:
# Paper mapping: Stage II, Progressive PMCG Distillation with Targeted Questioning; Figure 2. Builds and refreshes the patient-specific medication conflict graph.
# This documentation annotation does not alter executable behavior.


from ..system_impl.agents_impl import EvidenceGraphBuilder

class PMCGBuilder:
    name = 'PMCGBuilder'
    def __init__(self):
        self.impl = EvidenceGraphBuilder()
    def run(self, state, rules=None):
        return self.impl.run(state)
    def refresh_after_audits(self, state, rules=None):
        return self.impl.refresh_after_audits(state)
