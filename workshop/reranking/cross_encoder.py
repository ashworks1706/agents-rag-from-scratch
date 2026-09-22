"""Cross-encoder reranking: rescore candidate chunks by reading the query and chunk together, which is more accurate than bi-encoder search. Use it to reorder a small first-stage candidate set."""
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, k=3):
        chunks = list(chunks)
        scores = self.model.predict([(query, c) for c in chunks])
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
        return [(chunks[i], float(scores[i])) for i in order]
