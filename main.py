import pandas as pd

df = pd.read_csv("data/amazon_labeled.csv")

print(df["intent"].value_counts())
print("\nMissing:", df["intent"].isna().sum())