"""Ensemble search: run several retrievers and combine them with RRF.

Like RRF, but framed as a general ensemble: give it any list of retriever
objects (each with a .search(query, k) method) and it fuses their rankings.

pip install sentence-transformers rank-bm25 numpy
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searching.semantic_topk import SemanticSearch
from searching.bm25_search import BM25Search

RRF_K = 60


class EnsembleSearch:
    def __init__(self, chunks, retrievers=None):
        self.chunks = list(chunks)
        self.retrievers = retrievers or [SemanticSearch(self.chunks), BM25Search(self.chunks)]

    def search(self, query, k=3, pool=10):
        fused = {}
        for retriever in self.retrievers:
            results = retriever.search(query, k=pool)
            for rank, (chunk, _score) in enumerate(results, start=1):
                fused[chunk] = fused.get(chunk, 0.0) + 1.0 / (RRF_K + rank)
        order = sorted(fused, key=lambda c: -fused[c])[:k]
        return [(chunk, float(fused[chunk])) for chunk in order]


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    for chunk, score in EnsembleSearch(chunks).search("how much are dues", k=2):
        print(f"{score:.4f}", chunk)
