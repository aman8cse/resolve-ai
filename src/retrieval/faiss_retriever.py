import pickle

import faiss
from sentence_transformers import SentenceTransformer


INDEX_PATH = "data/artifacts/retrieval/amazon.index"
METADATA_PATH = "data/artifacts/retrieval/metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


class FAISSRetriever:

    def __init__(
        self,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        model_name=MODEL_NAME,
    ):
        print("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

        print("Loading FAISS index...")

        self.index = faiss.read_index(index_path)

        print("Loading metadata...")

        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index/metadata mismatch: "
                f"{self.index.ntotal} vectors vs "
                f"{len(self.metadata)} metadata rows"
            )

        print(
            f"Retriever ready: {self.index.ntotal} documents"
        )

    def retrieve(self, query, k=5):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        similarities, indices = self.index.search(
            query_embedding,
            k,
        )

        results = []

        for similarity, idx in zip(
            similarities[0],
            indices[0],
        ):
            if idx == -1:
                continue

            row = self.metadata.iloc[int(idx)]

            results.append({
                "similarity": float(similarity),
                "tweet_id": int(row["tweet_id_customer"]),
                "customer": row["text_customer"],
                "response": row["text_amazon"],
            })

        return results


def main():

    retriever = FAISSRetriever()

    queries = [
        "My package says delivered but I never received it.",
        "I haven't received my parcel.",
        "Someone hacked my Amazon account.",
        "When do I have to pay for my Amazon order?",
    ]

    for query in queries:

        print("\n" + "=" * 80)
        print("QUERY")
        print("=" * 80)
        print(query)

        results = retriever.retrieve(
            query,
            k=5,
        )

        for i, result in enumerate(
            results,
            start=1,
        ):
            print(f"\n--- Result {i} ---")
            print(
                "Similarity:",
                round(result["similarity"], 4),
            )
            print(
                "Customer:",
                result["customer"],
            )
            print(
                "Amazon:",
                result["response"],
            )


if __name__ == "__main__":
    main()