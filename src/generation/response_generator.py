import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


class ResponseGenerator:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.client = Groq(
            api_key=api_key
        )

    def generate(self, customer_message, intent, retrieved_cases):

        examples = "\n\n".join(
            [
                f"Customer: {case['customer']}\n"
                f"Amazon response: {case['response']}"
                for case in retrieved_cases
            ]
        )

        prompt = f"""
You are an Amazon customer-support reply assistant.

Your job is to draft a short, helpful response to the customer.

Customer message:
{customer_message}

Detected intent:
{intent}

Historical Amazon support examples:
{examples}

Instructions:
- Use the historical examples as evidence for how Amazon handled similar issues.
- Do not invent policies, refunds, compensation, or guarantees.
- Do not expose internal information.
- Do not ask for sensitive information publicly.
- Keep the response concise and professional.
- If the issue requires account-specific investigation, direct the customer to official support.
- Return ONLY the proposed customer-facing reply.
"""

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="openai/gpt-oss-120b",
            )

            return response.choices[0].message.content

        except Exception as e:
            print("LLM generation failed:", e)

            if retrieved_cases:
                return retrieved_cases[0]["response"]

            return "I'm sorry you're experiencing this issue. Please contact customer support for further assistance."