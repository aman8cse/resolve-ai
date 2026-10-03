import time

from src.models.train_classifier import IntentClassifier
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.qdrant_retriever import QdrantRetriever
from src.generation.response_generator import ResponseGenerator
from src.decision.escalation import EscalationDecision


class SupportAgent:

    def __init__(self):
        print("Loading intent classifier...")
        self.classifier = IntentClassifier()

        print("Loading historical retriever...")
        self.retriever = QdrantRetriever()

        print("Loading response generator...")
        self.generator = ResponseGenerator()

        print("Loading escalation")
        self.escalation = EscalationDecision()

    def analyze(self, message):

        start = time.perf_counter()

        classifier_start = time.perf_counter()

        classification = self.classifier.predict(message)

        intent = classification["intent"]
        intent_confidence = classification["confidence"]

        classifier_latency = time.perf_counter() - classifier_start

        retrieval_start = time.perf_counter()

        cases = self.retriever.retrieve(
            message,
            k=5
        )

        retrieval_latency = time.perf_counter() - retrieval_start

        generation_start = time.perf_counter()

        reply = self.generator.generate(
            customer_message=message,
            intent=intent,
            retrieved_cases=cases
        )

        generation_latency = time.perf_counter() - generation_start

        escalation_start = time.perf_counter()

        decision = self.escalation.decide(
            intent=intent,
            response=reply
        )

        escalation_latency = time.perf_counter() - escalation_start

        total_latency = time.perf_counter() - start

        return {
            "message": message,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "cases": cases,
            "reply": reply,
            "decision": decision["decision"],
            "reason": decision["reason"],

            "latency": {
                "classifier_ms": round(
                    classifier_latency * 1000, 2
                ),
                "retrieval_ms": round(
                    retrieval_latency * 1000, 2
                ),
                "generation_ms": round(
                    generation_latency * 1000, 2
                ),
                "escalation_ms": round(
                    escalation_latency * 1000, 2
                ),
                "total_ms": round(
                    total_latency * 1000, 2
                )
            }
        }


if __name__ == "__main__":

    agent = SupportAgent()

    result = agent.analyze(
        "My package says delivered but I never received it."
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