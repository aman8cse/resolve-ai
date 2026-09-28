from src.models.train_classifier import IntentClassifier
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.generation.response_generator import ResponseGenerator
from src.decision.escalation import EscalationDecision


class SupportAgent:

    def __init__(self):
        print("Loading intent classifier...")
        self.classifier = IntentClassifier()

        print("Loading historical retriever...")
        self.retriever = TfidfRetriever(
            "data/amazon_pairs.csv"
        )

        print("Loading response generator...")
        self.generator = ResponseGenerator()

        print("Loading escalation")
        self.escalation = EscalationDecision()

    def analyze(self, message):

        intent = self.classifier.predict(message)

        cases = self.retriever.retrieve(
            message,
            k=5
        )

        reply = self.generator.generate(
            customer_message=message,
            intent=intent,
            retrieved_cases=cases
        )

        decision = self.escalation.decide(
            intent=intent,
            response=reply
        )

        return {
            "message": message,
            "intent": intent,
            "cases": cases,
            "reply": reply,
            "decision": decision["decision"],
            "reason": decision["reason"],
        }


if __name__ == "__main__":

    agent = SupportAgent()

    result = agent.analyze(
        "Someone hacked my amazon account."
    )

    print("\n==============================")
    print("CUSTOMER")
    print("==============================")
    print(result["message"])

    print("\n==============================")
    print("INTENT")
    print("==============================")
    print(result["intent"])

    print("\n==============================")
    print("DECISION")
    print("==============================")
    print(result["decision"])

    print("\n==============================")
    print("REASON")
    print("==============================")
    print(result["reason"])

    print("\n==============================")
    print("HISTORICAL CASES")
    print("==============================")

    for i, case in enumerate(result["cases"], 1):

        print(f"\n--- Case {i} ---")
        print("Similarity:", round(case["similarity"], 4))
        print("Customer:", case["customer"])
        print("Amazon:", case["response"])

    print("\n==============================")
    print("GENERATED REPLY")
    print("==============================")
    print(result["reply"])