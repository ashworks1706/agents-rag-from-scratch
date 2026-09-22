"""FAISS index: fast similarity search with an in-memory index.

Uses IndexFlatIP (inner product). With L2-normalised vectors, inner product is
cosine similarity. FAISS is a library, not a server, so there is no setup.

pip install faiss-cpu numpy
"""
import numpy as np
import faiss


class FaissIndex:
    def __init__(self, vectors):
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def query(self, query_vector, k=3):
        q = np.asarray([query_vector], dtype="float32")
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(5, 8)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    idx = FaissIndex(vecs)
    print(idx.query(vecs[2], k=3))
