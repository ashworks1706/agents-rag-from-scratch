"""Ensemble search: run several retrievers and combine their rankings with RRF. Give it any objects that have a .search(query, k) method."""
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
            for rank, (chunk, _score) in enumerate(retriever.search(query, k=pool), start=1):
                fused[chunk] = fused.get(chunk, 0.0) + 1.0 / (RRF_K + rank)
        order = sorted(fused, key=lambda c: -fused[c])[:k]
        return [(chunk, float(fused[chunk])) for chunk in order]
