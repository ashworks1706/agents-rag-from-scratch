"""HNSW index: approximate nearest-neighbour search with a graph.

HNSW builds a navigable graph over the vectors, giving fast approximate search
that scales to millions of vectors. Uses the cosine space here.

pip install hnswlib numpy
"""
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
        # cosine distance -> similarity
        return [(int(i), float(1.0 - d)) for i, d in zip(ids[0], distances[0])]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(6, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = HnswIndex(vecs)
    print(idx.query(vecs[2], k=3))
