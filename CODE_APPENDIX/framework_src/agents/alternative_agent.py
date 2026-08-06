# AAAI-27 paper reference:
# Paper mapping: Decision Synthesis and Verification Layer; Stage III. Implements the paper-visible Alternative Agent.
# This documentation annotation does not alter executable behavior.


from ..system_impl.agents_impl import SaferAlternativeAgent

class AlternativeAgent:
    name = 'AlternativeAgent'
    def __init__(self):
        self.impl = SaferAlternativeAgent()
    def run(self, state, rules=None):
        if not state.get('yes_no_decision_board'):
            state.setdefault('blackboard', {})['alternative_search_deferred'] = True
            state.setdefault('messages', []).append({'agent': self.name, 'status': 'deferred', 'summary': 'Deferred alternative selection until yes/no safety gates are available.'})
            return state
        return self.impl.run(state)
