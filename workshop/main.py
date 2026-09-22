"""End-to-end RAG pipeline composed from the modular workshop folders.

Each stage is a folder; each file in it is one method. This script picks one
method per stage and chains them:

    splitting -> embedding -> indexing -> searching -> reranking

To try a different method, change the import on the marked line. For example,
swap `indexing.numpy_flat` for `indexing.faiss_index`, or replace the manual
embed+index+search with one retriever from `searching/` (bm25, hybrid,
query_fusion, reciprocal_rank_fusion, ensemble, router).

Run from the workshop folder:  python main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- pick one method per stage (change these imports to experiment) ----------
from splitting import recursive as splitter          # splitting/recursive.py
from embedding.dense import DenseEmbedder            # embedding/dense.py
from indexing.numpy_flat import NumpyIndex           # indexing/numpy_flat.py
from reranking.cross_encoder import Reranker         # reranking/cross_encoder.py
# -----------------------------------------------------------------------------

DOC = "sample_document.pdf"


def load_pdf(path):
    import pymupdf
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def run(query, first_stage_k=8, final_k=3):
    text = load_pdf(DOC)

    # 1. splitting
    chunks = splitter.split(text, chunk_size=500, overlap=100)

    # 2. embedding
    embedder = DenseEmbedder()
    doc_vectors = embedder.embed(chunks)

    # 3. indexing
    index = NumpyIndex(doc_vectors)

    # 4. searching (first-stage retrieval)
    hits = index.query(embedder.embed_query(query), k=first_stage_k)
    candidates = [chunks[i] for i, _score in hits]

    # 5. reranking (reorder the candidates for precision)
    reranked = Reranker().rerank(query, candidates, k=final_k)

    print(f"Query: {query}\n")
    print(f"Indexed {len(chunks)} chunks; retrieved {len(candidates)}; kept top {final_k}.\n")
    for rank, (chunk, score) in enumerate(reranked, start=1):
        print(f"[{rank}] score={score:.3f}  {chunk[:100]}...")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "How much does membership cost?"
    run(query)
