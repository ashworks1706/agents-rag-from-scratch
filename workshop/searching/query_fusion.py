"""Query fusion: search with several phrasings of the query and average their scores for broader coverage. Pass your own variations, or a few simple ones are derived."""
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
