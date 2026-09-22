"""RAG pipeline: splitting -> embedding -> indexing -> searching -> reranking.

Change any import below to swap a method for another file in that folder.
Run from the workshop folder:  python main.py "your question"
"""

from utils import load_pdf, get_query, print_results

from splitting import recursive as splitting          # try: character, token_based, spacy_nlp
from embedding.dense import DenseEmbedder             # embedding method
from indexing.numpy_flat import NumpyIndex            # try: faiss_index, hnsw_index
from reranking.cross_encoder import Reranker          # reranking method

query = get_query(default="How much does membership cost?")

# 1. splitting
text = load_pdf("sample_document.pdf")
chunks = splitting.split(text, chunk_size=500, overlap=100)

# 2. embedding
embedder = DenseEmbedder()
vectors = embedder.embed(chunks)

# 3. indexing
index = NumpyIndex(vectors)

# 4. searching
hits = index.query(embedder.embed_query(query), k=8)
candidates = [chunks[i] for i, score in hits]

# 5. reranking
results = Reranker().rerank(query, candidates, k=3)

print_results(query, chunks, results)
