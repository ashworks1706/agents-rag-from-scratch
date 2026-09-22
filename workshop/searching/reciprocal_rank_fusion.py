"""Reciprocal Rank Fusion (RRF): merge several rankings by rank position.

RRF combines ranked lists from different retrievers without comparing their raw
scores. Each document scores sum(1 / (k + rank)) across the lists. Here we fuse a
dense ranking and a BM25 ranking.

pip install sentence-transformers rank-bm25 numpy
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from embedding.dense import DenseEmbedder
from embedding.sparse_bm25 import BM25Model

RRF_K = 60


def _ranking(scores):
    return list(np.argsort(-np.asarray(scores)))


def rrf_fuse(rankings, k=RRF_K):
    fused = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class RRFSearch:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.embedder = DenseEmbedder()
        self.doc_vecs = self.embedder.embed(self.chunks)
        self.bm25 = BM25Model(self.chunks)

    def search(self, query, k=3):
        dense_rank = _ranking(self.doc_vecs @ self.embedder.embed_query(query))
        bm25_rank = _ranking(self.bm25.scores(query))
        fused = rrf_fuse([dense_rank, bm25_rank])
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [(self.chunks[i], float(fused[i])) for i in order]


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    for chunk, score in RRFSearch(chunks).search("cost of dues on tuesday", k=3):
        print(f"{score:.4f}", chunk)
