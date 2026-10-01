import pandas as pd
import re

DATA_PATH = "data/amazon_pairs.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDtypes:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())


# --------------------------------------------------
# DUPLICATES
# --------------------------------------------------

print("\n" + "=" * 80)
print("DUPLICATES")
print("=" * 80)

print("Exact duplicate rows:", df.duplicated().sum())

print(
    "Duplicate customer messages:",
    df["text_customer"].duplicated().sum()
)

print(
    "Duplicate customer-response pairs:",
    df.duplicated(
        subset=["text_customer", "text_amazon"]
    ).sum()
)

print(
    "Unique customer messages:",
    df["text_customer"].nunique()
)

print(
    "Unique Amazon responses:",
    df["text_amazon"].nunique()
)


# --------------------------------------------------
# TEXT LENGTH
# --------------------------------------------------

df["customer_length"] = (
    df["text_customer"]
    .fillna("")
    .str.len()
)

df["response_length"] = (
    df["text_amazon"]
    .fillna("")
    .str.len()
)

df["customer_words"] = (
    df["text_customer"]
    .fillna("")
    .str.split()
    .str.len()
)

df["response_words"] = (
    df["text_amazon"]
    .fillna("")
    .str.split()
    .str.len()
)


print("\n" + "=" * 80)
print("TEXT LENGTH")
print("=" * 80)

print("\nCustomer characters:")
print(df["customer_length"].describe())

print("\nAmazon response characters:")
print(df["response_length"].describe())

print("\nCustomer words:")
print(df["customer_words"].describe())

print("\nAmazon response words:")
print(df["response_words"].describe())


# --------------------------------------------------
# VERY SHORT RECORDS
# --------------------------------------------------

print("\n" + "=" * 80)
print("VERY SHORT RECORDS")
print("=" * 80)

print(
    "Customer messages <= 5 chars:",
    (df["customer_length"] <= 5).sum()
)

print(
    "Customer messages <= 10 chars:",
    (df["customer_length"] <= 10).sum()
)

print(
    "Responses <= 5 chars:",
    (df["response_length"] <= 5).sum()
)

print(
    "Responses <= 10 chars:",
    (df["response_length"] <= 10).sum()
)


# --------------------------------------------------
# URL / MENTION / HASHTAG
# --------------------------------------------------

customer_text = df["text_customer"].fillna("")

print("\n" + "=" * 80)
print("TEXT CHARACTERISTICS")
print("=" * 80)

print(
    "Contains URL:",
    customer_text.str.contains(
        r"http\S+",
        regex=True
    ).mean()
)

print(
    "Contains @mention:",
    customer_text.str.contains(
        r"@\w+",
        regex=True
    ).mean()
)

print(
    "Contains hashtag:",
    customer_text.str.contains(
        r"#\w+",
        regex=True
    ).mean()
)


# --------------------------------------------------
# MOST COMMON RESPONSES
# --------------------------------------------------

print("\n" + "=" * 80)
print("MOST COMMON AMAZON RESPONSES")
print("=" * 80)

response_counts = (
    df["text_amazon"]
    .value_counts()
)

for response, count in response_counts.head(20).items():
    print(f"\n[{count} occurrences]")
    print(response)


# --------------------------------------------------
# RANDOM EXAMPLES
# --------------------------------------------------

print("\n" + "=" * 80)
print("RANDOM EXAMPLES")
print("=" * 80)

sample = df.sample(20, random_state=42)

for _, row in sample.iterrows():
    print("\nCUSTOMER:")
    print(row["text_customer"])

    print("AMAZON:")
    print(row["text_amazon"])

    print("-" * 100)