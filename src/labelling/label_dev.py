import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai

from src.taxonomy.intents import INTENTS


INPUT_PATH = "data/amazon_sample.csv"
OUTPUT_PATH = "data/amazon_labeled.csv"

BATCH_SIZE = 50
MODEL = "gemini-3-flash-preview"

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def build_prompt(messages):
    intent_text = "\n".join(
        f"- {name}: {description}"
        for name, description in INTENTS.items()
    )

    numbered_messages = "\n".join(
        f"{i}: {message}"
        for i, message in enumerate(messages)
    )

    return f"""
You are labeling Amazon customer-support messages.

Assign exactly ONE intent to each message.

Available intents:

{intent_text}

Rules:
- Choose the most specific applicable intent.
- Do not invent new intents.
- Use "other" when none fits confidently.
- Return ONLY valid JSON.
- Return this exact structure:

{{
  "labels": [
    {{"index": 0, "intent": "delivery_issue"}},
    {{"index": 1, "intent": "payment_issue"}}
  ]
}}

Messages:

{numbered_messages}
"""


def label_batch(messages):
    response = client.models.generate_content(
        model=MODEL,
        contents=build_prompt(messages)
    )

    text = response.text.strip()

    # Remove markdown fences if the model adds them
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    return json.loads(text)


def main():
    df = pd.read_csv(INPUT_PATH)

    results = []

    for start in range(0, len(df), BATCH_SIZE):

        batch = df.iloc[start:start + BATCH_SIZE]

        print(
            f"Labeling {start + 1}-"
            f"{min(start + BATCH_SIZE, len(df))}"
        )

        try:
            output = label_batch(
                batch["text_customer"].tolist()
            )

            for item in output["labels"]:
                results.append({
                    "row_index": start + item["index"],
                    "intent": item["intent"]
                })

        except Exception as e:
            print("Error:", e)

        time.sleep(1)

    labels = pd.DataFrame(results)

    df["row_index"] = df.index

    df = df.merge(
        labels,
        on="row_index",
        how="left"
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFinished.")
    print("Rows:", len(df))
    print(
        "Labels:",
        df["intent"].notna().sum()
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()