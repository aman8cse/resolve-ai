from src.agent import SupportAgent


TEST_CASES = [
    {
        "message": "My package says delivered but I never received it.",
        "expected_intent": "delivery_issue",
    },
    {
        "message": "Someone hacked my Amazon account and changed my password.",
        "expected_intent": "account_issue",
    },
    {
        "message": "When will I be charged for my Amazon order?",
        "expected_intent": "payment_issue",
    },
    {
        "message": "I want to return the item I bought.",
        "expected_intent": "return_issue",
    },
    {
        "message": "I want a refund for my order.",
        "expected_intent": "refund_issue",
    },
    {
        "message": "Amazon Prime Video isn't working.",
        "expected_intent": "digital_content_issue",
    },
    {
        "message": "Where is my order?",
        "expected_intent": "order_issue",
    },
]


def main():

    agent = SupportAgent()

    correct = 0

    print("\n===================================")
    print("END-TO-END AGENT EVALUATION")
    print("===================================")

    for i, case in enumerate(TEST_CASES, start=1):

        result = agent.analyze(case["message"])

        predicted = result["intent"]
        expected = case["expected_intent"]

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        print(f"\n--- Test Case {i} ---")
        print("Message:", case["message"])
        print("Expected:", expected)
        print("Predicted:", predicted)
        print("Result:", "✓ PASS" if is_correct else "✗ FAIL")

        print("\nGenerated response:")
        print(result["reply"])

    accuracy = correct / len(TEST_CASES)

    print("\n===================================")
    print("RESULT")
    print("===================================")
    print(f"Correct: {correct}/{len(TEST_CASES)}")
    print(f"Intent accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()