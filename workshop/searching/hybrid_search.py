"""Hybrid search: mix dense (semantic) and BM25 (keyword) scores.

Combines the strengths of both: semantic matches paraphrases, BM25 catches exact
terms. alpha controls the balance (1.0 dense, 0.0 BM25).

pip install sentence-transformers rank-bm25 numpy
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.hybrid import HybridScorer


class HybridSearch:
    def __init__(self, chunks, alpha=0.5):
        self.chunks = list(chunks)
        self.scorer = HybridScorer(self.chunks, alpha=alpha)

    def search(self, query, k=3):
        return [(self.chunks[i], score) for i, score in self.scorer.top_k(query, k=k)]


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    for chunk, score in HybridSearch(chunks, alpha=0.5).search("how much are dues", k=2):
        print(f"{score:.3f}", chunk)
