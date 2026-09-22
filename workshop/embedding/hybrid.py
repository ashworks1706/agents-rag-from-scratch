"""Hybrid: combine dense (semantic) and BM25 (keyword) scores. Each is min-max normalised to [0, 1] and mixed by alpha (1=dense, 0=BM25)."""
import numpy as np
from embedding.dense import DenseEmbedder
from embedding.sparse_bm25 import BM25Model


def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class HybridScorer:
    def __init__(self, corpus, alpha=0.5):
        self.corpus = list(corpus)
        self.alpha = alpha
        self.embedder = DenseEmbedder()
        self.doc_vecs = self.embedder.embed(self.corpus)
        self.bm25 = BM25Model(self.corpus)

    def top_k(self, query, k=3):
        dense = self.doc_vecs @ self.embedder.embed_query(query)
        sparse = self.bm25.scores(query)
        combined = self.alpha * _minmax(dense) + (1 - self.alpha) * _minmax(sparse)
        order = np.argsort(-combined)[:k]
        return [(int(i), float(combined[i])) for i in order]
