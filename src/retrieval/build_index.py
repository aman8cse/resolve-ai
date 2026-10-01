import os
import pickle
import time

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer


DATA_PATH = "data/amazon_retrieval_corpus.csv"
INDEX_DIR = "data/artifacts/retrieval"

MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    start = time.perf_counter()

    print("Loading corpus...")
    df = pd.read_csv(DATA_PATH)

    texts = df["customer_for_retrieval"].tolist()

    print(f"Documents: {len(texts)}")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Embedding shape:", embeddings.shape)

    # Build exact cosine-similarity index.
    # With normalized vectors, inner product == cosine similarity.
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("FAISS vectors:", index.ntotal)

    index_path = os.path.join(
        INDEX_DIR,
        "amazon.index"
    )

    metadata_path = os.path.join(
        INDEX_DIR,
        "metadata.pkl"
    )

    config_path = os.path.join(
        INDEX_DIR,
        "config.pkl"
    )

    faiss.write_index(index, index_path)

    metadata = df[
        [
            "tweet_id_customer",
            "text_customer",
            "text_amazon",
            "created_at_customer",
            "created_at_amazon",
        ]
    ]

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    config = {
        "model_name": MODEL_NAME,
        "documents": len(df),
        "embedding_dimension": dimension,
        "normalized": True,
    }

    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"Documents: {len(df)}")
    print(f"Dimensions: {dimension}")
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Time: {elapsed / 60:.2f} minutes")
    print(f"Index: {index_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()