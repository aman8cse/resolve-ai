import pandas as pd
import re

INPUT_PATH = "data/amazon_pairs.csv"
OUTPUT_PATH = "data/amazon_retrieval_corpus.csv"


def normalize_customer_text(text):
    text = str(text).strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Normalize Twitter user IDs
    text = re.sub(r"@\d+", "@USER", text)

    # Normalize URLs
    text = re.sub(r"https?://\S+", "URL", text)

    return text


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Original rows:", len(df))

    # Remove missing values defensively
    df = df.dropna(
        subset=["text_customer", "text_amazon"]
    ).copy()

    # Normalize only the retrieval representation
    df["customer_for_retrieval"] = (
        df["text_customer"]
        .apply(normalize_customer_text)
    )

    # Remove empty retrieval texts
    df = df[
        df["customer_for_retrieval"].str.len() > 0
    ].copy()

    # Remove exact duplicate customer-response pairs
    df = df.drop_duplicates(
        subset=["text_customer", "text_amazon"]
    ).copy()

    columns = [
        "tweet_id_customer",
        "text_customer",
        "text_amazon",
        "created_at_customer",
        "created_at_amazon",
        "customer_for_retrieval",
    ]

    df = df[columns]

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Final rows:", len(df))
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()