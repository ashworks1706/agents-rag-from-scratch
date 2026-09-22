"""Chroma index: an embedded vector database that stores text and vectors together and handles search, in-process with no server."""
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
