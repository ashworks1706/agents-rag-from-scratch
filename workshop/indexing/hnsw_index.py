"""HNSW index: approximate nearest-neighbour search over a navigable graph. Fast and scales to millions of vectors; uses the cosine space."""
import numpy as np
import hnswlib


class HnswIndex:
    def __init__(self, vectors, ef=50, M=16):
        vectors = np.asarray(vectors, dtype="float32")
        n, dim = vectors.shape
        self.index = hnswlib.Index(space="cosine", dim=dim)
        self.index.init_index(max_elements=n, ef_construction=200, M=M)
        self.index.add_items(vectors, np.arange(n))
        self.index.set_ef(ef)

    def query(self, query_vector, k=3):
        q = np.asarray([query_vector], dtype="float32")
        ids, distances = self.index.knn_query(q, k=k)
        return [(int(i), float(1.0 - d)) for i, d in zip(ids[0], distances[0])]
