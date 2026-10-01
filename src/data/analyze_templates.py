import pandas as pd
import re

DATA_PATH = "data/amazon_pairs.csv"


def normalize_response(text):
    text = str(text)

    # Remove @user references
    text = re.sub(r"@\d+", "@USER", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", "URL", text)

    # Remove agent signature such as ^VB, ^EM, ^PS
    text = re.sub(r"\^[A-Z]{2}\b", "^AGENT", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    df = pd.read_csv(DATA_PATH)

    df["normalized_response"] = (
        df["text_amazon"]
        .fillna("")
        .apply(normalize_response)
    )

    print("=" * 80)
    print("RESPONSE TEMPLATE ANALYSIS")
    print("=" * 80)

    print("Original responses:", len(df))

    unique_original = df["text_amazon"].nunique()
    unique_normalized = df["normalized_response"].nunique()

    print("Unique original responses:", unique_original)
    print("Unique normalized responses:", unique_normalized)

    reduction = 1 - (
        unique_normalized / unique_original
    )

    print(
        f"Template reduction: {reduction:.2%}"
    )

    print("\nTop normalized templates:")

    counts = (
        df["normalized_response"]
        .value_counts()
        .head(20)
    )

    for template, count in counts.items():
        print(f"\n[{count} occurrences]")
        print(template)


if __name__ == "__main__":
    main()