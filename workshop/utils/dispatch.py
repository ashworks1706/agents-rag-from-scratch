"""Maps method names to the modules in each stage folder.

Used by the benchmark harness so it can select a splitter or search method by
name. main.py does not use this; it imports the methods it wants directly.
"""


def get_chunks(splitter, text, chunk_size, overlap):
    if splitter in ("recursive", "character", "token_based"):
        mod = __import__(f"splitting.{splitter}", fromlist=["split"])
        return mod.split(text, chunk_size=chunk_size, overlap=overlap)
    if splitter in ("sentence_nltk", "spacy_nlp"):
        mod = __import__(f"splitting.{splitter}", fromlist=["split"])
        return mod.split(text, chunk_size=chunk_size)
    mod = __import__(f"splitting.{splitter}", fromlist=["split"])
    return mod.split(text)


def build_search(search, chunks):
    if search == "semantic":
        from searching.semantic_topk import SemanticSearch
        return SemanticSearch(chunks)
    if search == "bm25":
        from searching.bm25_search import BM25Search
        return BM25Search(chunks)
    if search == "hybrid":
        from searching.hybrid_search import HybridSearch
        return HybridSearch(chunks)
    if search == "query_fusion":
        from searching.query_fusion import QueryFusionSearch
        return QueryFusionSearch(chunks)
    if search == "rrf":
        from searching.reciprocal_rank_fusion import RRFSearch
        return RRFSearch(chunks)
    if search == "ensemble":
        from searching.ensemble import EnsembleSearch
        return EnsembleSearch(chunks)
    if search == "router":
        from searching.router import RouterSearch
        return RouterSearch(chunks)
    raise ValueError(f"unknown search method: {search}")


def retrieve(searcher, query, k):
    results = searcher.search(query, k=k)
    if isinstance(results, tuple):  # router returns (choice, results)
        results = results[1]
    return [chunk for chunk, _score in results]
