import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_PATH = "data/amazon_pairs.csv"


class EmbeddingRetriever:

    def __init__(self, data_path):

        self.df = pd.read_csv(data_path)

        self.df = self.df.dropna(
            subset=["text_customer", "text_amazon"]
        ).reset_index(drop=True)

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Creating embeddings...")

        self.customer_embeddings = self.model.encode(
            self.df["text_customer"].tolist(),
            show_progress_bar=True,
            normalize_embeddings=True
        )

    def retrieve(
        self,
        query,
        k=5,
        exclude_tweet_id=None
    ):

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        similarities = cosine_similarity(
            query_embedding,
            self.customer_embeddings
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
                "tweet_id": row["tweet_id_customer"],
                "customer": row["text_customer"],
                "response": row["text_amazon"]
            })

        return results


def main():

    retriever = EmbeddingRetriever(INPUT_PATH)

    queries = [
        "My package says delivered but I never received it.",
        "I haven't received my parcel.",
        "Someone hacked my Amazon account.",
        "When do I have to pay for my Amazon order?"
    ]

    for query in queries:

        print("\n===================================")
        print("QUERY")
        print("===================================")
        print(query)

        results = retriever.retrieve(query, k=5)

        for i, result in enumerate(results, start=1):

            print(f"\n--- Result {i} ---")
            print(
                "Similarity:",
                round(result["similarity"], 4)
            )
            print("Customer:", result["customer"])
            print("Amazon:", result["response"])


if __name__ == "__main__":
    main()