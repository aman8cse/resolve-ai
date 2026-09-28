import pandas as pd

INPUT_PATH = "data/amazon_pairs.csv"
OUTPUT_PATH = "data/amazon_sample.csv"

SAMPLE_SIZE = 2_000
RANDOM_STATE = 42

df = pd.read_csv(INPUT_PATH)

sample = df.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_STATE
)

sample.to_csv(OUTPUT_PATH, index=False)

print("Original dataset:", len(df))
print("Sample size:", len(sample))
print("Saved to:", OUTPUT_PATH)