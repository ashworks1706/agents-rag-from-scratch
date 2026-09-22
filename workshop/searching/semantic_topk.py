"""Semantic top-k search: embed the query and return the closest chunks.

The standard dense retriever. Embeds the chunks once, then for a query embeds it
and returns the k chunks with the highest cosine similarity.

pip install sentence-transformers numpy
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.dense import DenseEmbedder
from indexing.numpy_flat import NumpyIndex


class SemanticSearch:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.embedder = DenseEmbedder()
        self.index = NumpyIndex(self.embedder.embed(self.chunks))

    def search(self, query, k=3):
        hits = self.index.query(self.embedder.embed_query(query), k=k)
        return [(self.chunks[i], score) for i, score in hits]


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    for chunk, score in SemanticSearch(chunks).search("how much are dues", k=2):
        print(f"{score:.3f}", chunk)
