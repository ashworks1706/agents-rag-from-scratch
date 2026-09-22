"""LSH index: approximate search with random-projection hashing.

Locality-sensitive hashing maps similar vectors to the same bucket using random
hyperplanes, so a query only compares against vectors in matching buckets. This
is a compact, from-scratch version for teaching.

pip install numpy
"""
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
        candidates = self.buckets.get(self._hash(q))
        if not candidates:  # fall back to all vectors if the bucket is empty
            candidates = range(len(self.vectors))
        scored = [(int(i), float(self.vectors[i] @ q)) for i in candidates]
        scored.sort(key=lambda t: -t[1])
        return scored[:k]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(20, 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = LSHIndex(vecs, n_bits=6)
    print(idx.query(vecs[3], k=3))
