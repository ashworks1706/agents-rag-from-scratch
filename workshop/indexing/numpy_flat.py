"""Flat NumPy index: brute-force exact cosine search. Scores every stored vector with one matrix multiply. Simple; fine up to a few hundred thousand vectors."""
import numpy as np


class NumpyIndex:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=float)

    def query(self, query_vector, k=3):
        scores = self.vectors @ np.asarray(query_vector, dtype=float)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]
