# AAAI-27 paper reference:
# Paper mapping: internal implementation detail supporting Stages II-III candidate construction. It is not presented as a separate paper-visible agent.
# This documentation annotation does not alter executable behavior.


from ..system_impl.agents_impl import TherapeuticNeedIdentifier, OpenCandidateGenerator

class GuidelineCandidateAgent:
    name = 'GuidelineCandidateAgent'
    def __init__(self):
        self.need = TherapeuticNeedIdentifier()
        self.generator = OpenCandidateGenerator()
    def run(self, state, rules=None):
        state = self.need.run(state)
        state = self.generator.run(state)
        return state
