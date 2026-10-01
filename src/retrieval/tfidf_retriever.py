import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = "data/amazon_pairs.csv"


class TfidfRetriever:

    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)

        self.df = self.df.dropna(
            subset=["text_customer", "text_amazon"]
        ).reset_index(drop=True)

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
        )

        self.customer_vectors = self.vectorizer.fit_transform(
            self.df["text_customer"]
        )

    def retrieve(self, query, k=5, exclude_tweet_id=None):

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.customer_vectors
        )[0]

        if exclude_tweet_id is not None:
            excluded = self.df[
                self.df["tweet_id_customer"] == exclude_tweet_id
            ].index

            similarities[excluded] = -1

        top_indices = similarities.argsort()[-k:][::-1]

        results = []

        for idx in top_indices:
            row = self.df.iloc[idx]

            results.append({
                "similarity": float(similarities[idx]),
                "tweet_id": int(row["tweet_id_customer"]),
                "customer": row["text_customer"],
                "response": row["text_amazon"]
            })

        return results


def main():

    retriever = TfidfRetriever(INPUT_PATH)

    query = "My package says delivered but I never received it."

    results = retriever.retrieve(query, k=5)

    print("\nQUERY:")
    print(query)

    print("\nTOP RESULTS:")

    for i, result in enumerate(results, start=1):

        print(f"\n--- Result {i} ---")
        print("Similarity:", round(result["similarity"], 4))
        print("Customer:", result["customer"])
        print("Amazon:", result["response"])


if __name__ == "__main__":
    main()