"""BM25 search: keyword retrieval with no neural model. Ranks chunks by BM25 score against the query."""
from embedding.sparse_bm25 import BM25Model


class BM25Search:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.model = BM25Model(self.chunks)

    def search(self, query, k=3):
        return [(self.chunks[i], score) for i, score in self.model.top_k(query, k=k)]
