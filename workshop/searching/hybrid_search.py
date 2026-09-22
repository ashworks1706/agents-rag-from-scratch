"""Hybrid search: mix dense (semantic) and BM25 (keyword) scores; alpha balances them (1=dense, 0=BM25)."""
from embedding.hybrid import HybridScorer


class HybridSearch:
    def __init__(self, chunks, alpha=0.5):
        self.chunks = list(chunks)
        self.scorer = HybridScorer(self.chunks, alpha=alpha)

    def search(self, query, k=3):
        return [(self.chunks[i], score) for i, score in self.scorer.top_k(query, k=k)]
