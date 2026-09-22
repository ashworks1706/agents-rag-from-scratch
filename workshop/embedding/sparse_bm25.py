"""Sparse BM25: keyword scoring with no neural model. Ranks documents by how often the query's words appear, adjusted for length."""
from rank_bm25 import BM25Okapi


class BM25Model:
    def __init__(self, corpus):
        self.corpus = list(corpus)
        self.tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized)

    def scores(self, query):
        return self.bm25.get_scores(query.lower().split())

    def top_k(self, query, k=3):
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(i, float(scores[i])) for i in order]
