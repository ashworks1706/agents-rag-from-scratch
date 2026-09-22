"""Dense embeddings with Sentence Transformers (all-MiniLM-L6-v2): text becomes a fixed-size vector, L2-normalised so a dot product equals cosine similarity."""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class DenseEmbedder:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        vecs = self.model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vecs / norms

    def embed_query(self, text):
        return self.embed([text])[0]
