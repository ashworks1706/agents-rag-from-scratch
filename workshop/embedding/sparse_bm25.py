"""Sparse BM25 representation: keyword scoring, no neural model.

BM25 ranks documents by how often the query's words appear in them, adjusted
for document length. It captures exact terms that dense embeddings can miss.

pip install rank-bm25
"""
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


if __name__ == "__main__":
    corpus = ["membership costs 15 dollars per semester",
              "workshops run every tuesday",
              "officer elections happen once a year"]
    m = BM25Model(corpus)
    for i, s in m.top_k("membership cost", k=3):
        print(f"{s:.3f}", corpus[i])
