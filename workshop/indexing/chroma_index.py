"""Chroma index: an embedded vector database that stores text and vectors.

Chroma keeps documents and their embeddings together and handles search. It
runs in-process (in-memory here), so there is no server to start.

pip install chromadb
Note: on Streamlit Community Cloud, Chroma needs a pysqlite3 workaround; for the
deployed app this repo uses FAISS instead.
"""
import chromadb


class ChromaIndex:
    def __init__(self, documents, embeddings, name="workshop"):
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(name)
        except Exception:
            pass
        self.collection = self.client.create_collection(name)
        self.collection.add(
            ids=[str(i) for i in range(len(documents))],
            documents=list(documents),
            embeddings=[list(map(float, v)) for v in embeddings],
        )

    def query(self, query_vector, k=3):
        res = self.collection.query(query_embeddings=[list(map(float, query_vector))], n_results=k)
        ids = [int(x) for x in res["ids"][0]]
        dists = res["distances"][0]
        return [(i, float(1.0 - d)) for i, d in zip(ids, dists)]


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(4, 8))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    docs = [f"doc {i}" for i in range(4)]
    idx = ChromaIndex(docs, vecs)
    print(idx.query(vecs[1], k=3))
