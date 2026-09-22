"""Semantic top-k: embed the query and return the k closest chunks by cosine similarity. The standard dense retriever."""
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
