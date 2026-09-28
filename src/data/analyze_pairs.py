import pandas as pd

DATA_PATH = "data/amazon_pairs.csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nResponse length:")
print(df["text_amazon"].str.len().describe())

print("\nCustomer message length:")
print(df["text_customer"].str.len().describe())

print("\nRandom examples:")
sample = df.sample(50, random_state=42)

for i, row in sample.iterrows():
    print("\nCUSTOMER:")
    print(row["text_customer"])

    print("AMAZON:")
    print(row["text_amazon"])

    print("-" * 100)