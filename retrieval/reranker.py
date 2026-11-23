
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    ):

        print("Loading cross-encoder model...")

        self.model = CrossEncoder(
            model_name,
            device=device
        )

        print("Cross-encoder loaded successfully!")

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        pairs = [
            (query, document["content"])
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False
        )

        reranked = []

        for document, score in zip(documents, scores):

            result = document.copy()

            result["reranker_score"] = float(score)

            reranked.append(result)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True
        )

        return reranked[:top_k]
