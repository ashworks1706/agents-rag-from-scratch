"""Query fusion: search with several phrasings of the query and merge results.

One query can miss relevant chunks. Query fusion runs several variations and
averages their scores for broader coverage. Pass your own variations (an LLM can
generate them); otherwise a few simple ones are derived.

pip install sentence-transformers numpy
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from embedding.dense import DenseEmbedder


def default_variations(query):
    return [query, f"{query} overview", f"{query} details", f"what is {query}"]


class QueryFusionSearch:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.embedder = DenseEmbedder()
        self.doc_vecs = self.embedder.embed(self.chunks)

    def search(self, query, k=3, variations=None):
        variations = variations or default_variations(query)
        scores = np.zeros(len(self.chunks))
        for q in variations:
            scores += self.doc_vecs @ self.embedder.embed_query(q)
        scores /= len(variations)
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    for chunk, score in QueryFusionSearch(chunks).search("dues", k=2):
        print(f"{score:.3f}", chunk)
