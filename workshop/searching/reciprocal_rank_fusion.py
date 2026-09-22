"""Reciprocal Rank Fusion (RRF): merge a dense ranking and a BM25 ranking by rank position, scoring each document sum(1/(k+rank))."""
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
