from src.agent import SupportAgent


class AgentService:

    def __init__(self):
        self.agent = SupportAgent()

    def analyze(self, message: str):
        return self.agent.analyze(message)