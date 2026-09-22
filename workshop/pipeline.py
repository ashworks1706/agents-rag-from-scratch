"""Your RAG pipeline. Edit this file, save, and the app reruns with your changes.

Swap a method by changing its import line to another file in that folder:

  splitting/   recursive, character, token_based, sentence_nltk, spacy_nlp
  searching/   semantic_topk, bm25_search, hybrid_search, query_fusion,
               reciprocal_rank_fusion, ensemble, router
  reranking/   cross_encoder
"""

from splitting.recursive import split                        # try: character, token_based, sentence_nltk
from searching.semantic_topk import SemanticSearch as Search  # try: bm25_search.BM25Search, hybrid_search.HybridSearch
from reranking.cross_encoder import Reranker

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
USE_RERANKER = False


def build(text):
    """Split the document and index it. Returns (chunks, searcher)."""
    chunks = split(text, CHUNK_SIZE, OVERLAP)
    return chunks, Search(chunks)
