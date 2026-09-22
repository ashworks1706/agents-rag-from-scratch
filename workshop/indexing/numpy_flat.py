"""Flat NumPy index: brute-force exact cosine search.

Stores all vectors and, for a query, scores every one with a single matrix
multiply. Exact and simple; fine up to a few hundred thousand vectors.

pip install numpy
"""
import numpy as np


class NumpyIndex:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=float)

    def query(self, query_vector, k=3):
        scores = self.vectors @ np.asarray(query_vector, dtype=float)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = NumpyIndex(vecs)
    print(idx.query(vecs[2], k=3))  # vector 2 should rank first
