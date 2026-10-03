import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


load_dotenv()


class QdrantRetriever:
    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        collection_name=None,
    ):
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url:
            raise ValueError("QDRANT_URL is missing")

        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY is missing")

        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION", "amazon-support-cases")
        )

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        print("Connecting to Qdrant...")
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # Verify that the collection exists.
        collection = self.client.get_collection(self.collection_name)

        print(
            f"Connected to Qdrant collection "
            f"'{self.collection_name}'"
        )
        print(f"Points: {collection.points_count}")

    def retrieve(self, message, k=5):
        """
        Retrieve the most semantically similar historical
        customer-support cases.
        """

        # Generate embedding for ONLY the incoming query.
        query_vector = self.model.encode(
            message,
            normalize_embeddings=True,
        ).tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        results = []

        for point in response.points:
            payload = point.payload

            results.append(
                {
                    "similarity": float(point.score),
                    "tweet_id": int(payload["tweet_id_customer"]),
                    "customer": payload["text_customer"],
                    "response": payload["text_amazon"],
                }
            )

        return results