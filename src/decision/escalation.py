class EscalationDecision:

    HUMAN_INTENTS = {
        "account_issue",
        "complaint",
    }

    def decide(self, intent, response=None):
        if intent in self.HUMAN_INTENTS:
            return {
                "decision": "HUMAN",
                "reason": f"{intent} may require account-specific or sensitive investigation."
            }

        return {
            "decision": "AUTO",
            "reason": f"{intent} can generally be handled with a standard support response."
        }