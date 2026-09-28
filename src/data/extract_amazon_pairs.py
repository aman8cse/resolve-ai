import pandas as pd

DATA_PATH = "data/twcs.csv"
OUTPUT_PATH = "data/amazon_pairs.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# 1. Keep AmazonHelp's responses
amazon = df[df["author_id"] == "AmazonHelp"].copy()

# 2. Keep only responses that have a parent tweet
amazon = amazon[amazon["in_response_to_tweet_id"].notna()].copy()

# 3. Create a lookup table for all tweets
tweets = df[
    ["tweet_id", "author_id", "inbound", "text", "created_at"]
].copy()

tweets["tweet_id"] = tweets["tweet_id"].astype(str)
amazon["in_response_to_tweet_id"] = (
    amazon["in_response_to_tweet_id"].astype(int).astype(str)
)

# 4. Find the customer tweet that each Amazon response answered
pairs = amazon.merge(
    tweets,
    left_on="in_response_to_tweet_id",
    right_on="tweet_id",
    suffixes=("_amazon", "_customer")
)

# 5. Keep only actual inbound/customer messages
pairs = pairs[pairs["inbound_customer"] == True].copy()

# 6. Select useful columns
pairs = pairs[
    [
        "tweet_id_customer",
        "text_customer",
        "tweet_id_amazon",
        "text_amazon",
        "created_at_customer",
        "created_at_amazon",
    ]
]

print("\nExtracted pairs:", len(pairs))

print("\nSample:")
print(pairs.head(10).to_string(index=False))

pairs.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved to: {OUTPUT_PATH}")