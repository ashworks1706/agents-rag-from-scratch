"""FAISS index: fast in-memory similarity search (IndexFlatIP). With L2-normalised vectors, inner product is cosine similarity. A library, not a server."""
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
