"""Cross-encoder reranking: rescore candidate chunks against the query.

A bi-encoder (dense search) is fast but approximate. A cross-encoder reads the
query and a chunk together and scores their relevance directly, which is more
accurate. Use it to reorder a small candidate set from a first-stage retriever.

pip install sentence-transformers
"""
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


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    for chunk, score in Reranker().rerank("how much are dues", chunks, k=2):
        print(f"{score:.3f}", chunk)
