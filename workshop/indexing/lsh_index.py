"""LSH index: approximate search by hashing vectors into buckets with random hyperplanes, so a query only compares against same-bucket vectors."""
import numpy as np


class LSHIndex:
    def __init__(self, vectors, n_bits=8, seed=0):
        self.vectors = np.asarray(vectors, dtype=float)
        rng = np.random.default_rng(seed)
        self.planes = rng.normal(size=(n_bits, self.vectors.shape[1]))
        self.buckets = {}
        for i, v in enumerate(self.vectors):
            self.buckets.setdefault(self._hash(v), []).append(i)

    def _hash(self, v):
        return tuple((self.planes @ v) > 0)

    def query(self, query_vector, k=3):
        q = np.asarray(query_vector, dtype=float)
        candidates = self.buckets.get(self._hash(q)) or range(len(self.vectors))
        scored = [(int(i), float(self.vectors[i] @ q)) for i in candidates]
        scored.sort(key=lambda t: -t[1])
        return scored[:k]
