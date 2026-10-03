import os
import pickle
import faiss
import pandas as pd

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()

FAISS_INDEX_PATH = "data/artifacts/retrieval/amazon.index"
METADATA_PATH = "data/artifacts/retrieval/metadata.pkl"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "amazon-support-cases"
)

BATCH_SIZE = 512


def main():
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is missing from .env")

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is missing from .env")

    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_PATH)

    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")

    print("Loading metadata...")
    metadata = pickle.load(open(METADATA_PATH, "rb"))

    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("Expected metadata.pkl to contain a pandas DataFrame")

    if len(metadata) != index.ntotal:
        raise ValueError(
            f"Vector/metadata mismatch: "
            f"{index.ntotal} vectors vs {len(metadata)} metadata rows"
        )

    print(f"Metadata rows: {len(metadata)}")

    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    collection = client.get_collection(COLLECTION_NAME)

    print(f"Connected to collection: {COLLECTION_NAME}")
    print(f"Collection vectors: {collection.points_count}")

    total = index.ntotal

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        # Reconstruct the already-existing vectors from FAISS.
        vectors = index.reconstruct_n(start, end - start)

        points = []

        for offset, vector in enumerate(vectors):
            row_index = start + offset
            row = metadata.iloc[row_index]

            point = PointStruct(
                id=row_index,
                vector=vector.tolist(),
                payload={
                    "tweet_id_customer": int(row["tweet_id_customer"]),
                    "text_customer": str(row["text_customer"]),
                    "text_amazon": str(row["text_amazon"]),
                    "created_at_customer": str(row["created_at_customer"]),
                    "created_at_amazon": str(row["created_at_amazon"]),
                },
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        print(f"Uploaded {end}/{total}")

    print("\nMigration complete!")

    collection = client.get_collection(COLLECTION_NAME)
    print(f"Qdrant points: {collection.points_count}")


if __name__ == "__main__":
    main()