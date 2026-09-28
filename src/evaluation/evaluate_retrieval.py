import pandas as pd

from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.embedding_retriever import EmbeddingRetriever


LABELED_PATH = "data/amazon_sample_labeled.csv"
PAIRS_PATH = "data/amazon_pairs.csv"


def evaluate_retriever(retriever, queries, labeled_reference, name, k=5):

    hits_at_1 = 0
    hits_at_5 = 0

    total = len(queries)

    for _, row in queries.iterrows():

        query = row["text_customer"]
        expected_intent = row["intent"]
        query_id = row["tweet_id_customer"]

        results = retriever.retrieve(
            query,
            k=k,
            exclude_tweet_id=query_id
        )

        retrieved_intents = []

        for result in results:

            retrieved_id = result.get("tweet_id")

            if retrieved_id in labeled_reference:
                retrieved_intents.append(
                    labeled_reference[retrieved_id]
                )

        # Recall@1
        if len(retrieved_intents) > 0:
            if retrieved_intents[0] == expected_intent:
                hits_at_1 += 1

        # Recall@5
        if expected_intent in retrieved_intents:
            hits_at_5 += 1

    recall_at_1 = hits_at_1 / total
    recall_at_5 = hits_at_5 / total

    print(f"\n{name}")
    print("-" * 40)
    print(f"Recall@1: {recall_at_1:.4f}")
    print(f"Recall@5: {recall_at_5:.4f}")

    return recall_at_1, recall_at_5


def main():

    labeled = pd.read_csv(LABELED_PATH)

    # Remove rows without labels
    labeled = labeled.dropna(subset=["intent"])

    # Reproducible train/evaluation split
    reference, queries = (
        __import__("sklearn.model_selection", fromlist=["train_test_split"])
        .train_test_split(
            labeled,
            test_size=0.2,
            random_state=42,
            stratify=labeled["intent"]
        )
    )

    print("Reference examples:", len(reference))
    print("Evaluation queries:", len(queries))

    # ---------------------------------------------------------
    # Build lookup:
    # tweet_id -> intent
    # ---------------------------------------------------------

    labeled_reference = dict(
        zip(
            reference["tweet_id_customer"],
            reference["intent"]
        )
    )

    print("\nKnown reference intents:", len(labeled_reference))

    # ---------------------------------------------------------
    # TF-IDF
    # ---------------------------------------------------------

    print("\nLoading TF-IDF retriever...")

    tfidf_retriever = TfidfRetriever(PAIRS_PATH)

    tfidf_result = evaluate_retriever(
        tfidf_retriever,
        queries,
        labeled_reference,
        "TF-IDF",
        k=5
    )

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    print("\nLoading embedding retriever...")

    embedding_retriever = EmbeddingRetriever(PAIRS_PATH)

    embedding_result = evaluate_retriever(
        embedding_retriever,
        queries,
        labeled_reference,
        "EMBEDDINGS",
        k=5
    )

    # ---------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------

    print("\n")
    print("=" * 55)
    print("RETRIEVAL COMPARISON")
    print("=" * 55)

    print(
        f"{'Model':<20}"
        f"{'Recall@1':<15}"
        f"{'Recall@5':<15}"
    )

    print(
        f"{'TF-IDF':<20}"
        f"{tfidf_result[0]:<15.4f}"
        f"{tfidf_result[1]:<15.4f}"
    )

    print(
        f"{'Embeddings':<20}"
        f"{embedding_result[0]:<15.4f}"
        f"{embedding_result[1]:<15.4f}"
    )


if __name__ == "__main__":
    main()