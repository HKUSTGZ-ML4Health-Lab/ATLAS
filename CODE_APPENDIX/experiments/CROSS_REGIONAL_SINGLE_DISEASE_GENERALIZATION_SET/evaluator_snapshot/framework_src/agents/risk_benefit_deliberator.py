
from ..system_impl.agents_impl import (
    SafetyFirstCandidatePruner,
    YesNoDecisionAgent,
    SaferAlternativeAgent,
    RiskBenefitDeliberator as _RiskBenefitDeliberator,
)

class RiskBenefitDeliberator:
    name = 'RiskBenefitDeliberator'
    def __init__(self):
        self.pruner = SafetyFirstCandidatePruner()
        self.yesno = YesNoDecisionAgent()
        self.alt = SaferAlternativeAgent()
        self.impl = _RiskBenefitDeliberator()
    def run(self, state, rules=None):
        state = self.pruner.run(state)
        state = self.yesno.run(state)
        if not state.get('alternative_selection') or not state.get('alternative_selection', {}).get('safer_alternative'):
            state = self.alt.run(state)
        return self.impl.run(state)
