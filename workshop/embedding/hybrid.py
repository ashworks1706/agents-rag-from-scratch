"""Hybrid embeddings: combine dense (semantic) and sparse (BM25) scores.

Each method scores a corpus; the scores are min-max normalised to [0, 1] and
mixed with a weight alpha (alpha=1 is pure dense, alpha=0 is pure BM25).

pip install sentence-transformers rank-bm25
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


if __name__ == "__main__":
    corpus = ["membership costs 15 dollars per semester",
              "workshops run every tuesday from 6 to 8 pm",
              "officer elections happen once a year"]
    h = HybridScorer(corpus, alpha=0.5)
    for i, s in h.top_k("how much are dues", k=3):
        print(f"{s:.3f}", corpus[i])
