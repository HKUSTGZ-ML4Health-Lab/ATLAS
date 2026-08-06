
from ..system_impl.agents_impl import EvidenceGraphBuilder

class PMCGBuilder:
    name = 'PMCGBuilder'
    def __init__(self):
        self.impl = EvidenceGraphBuilder()
    def run(self, state, rules=None):
        return self.impl.run(state)
    def refresh_after_audits(self, state, rules=None):
        return self.impl.refresh_after_audits(state)
