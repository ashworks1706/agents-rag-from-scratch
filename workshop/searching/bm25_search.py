"""BM25 search: keyword retrieval with no neural model.

Good when exact words matter. Ranks chunks by BM25 score against the query.

pip install rank-bm25
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.sparse_bm25 import BM25Model


class BM25Search:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.model = BM25Model(self.chunks)

    def search(self, query, k=3):
        return [(self.chunks[i], score) for i, score in self.model.top_k(query, k=k)]


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    for chunk, score in BM25Search(chunks).search("workshop tuesday", k=2):
        print(f"{score:.3f}", chunk)
