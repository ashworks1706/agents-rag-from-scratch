"""Router search: send short, keyword-like queries to BM25 and longer, natural-language queries to semantic search. Swap the rule for an LLM classifier if you like."""
from searching.semantic_topk import SemanticSearch
from searching.bm25_search import BM25Search


class RouterSearch:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.semantic = SemanticSearch(self.chunks)
        self.bm25 = BM25Search(self.chunks)
        self.last_choice = None

    def route(self, query):
        return "bm25" if len(query.split()) <= 3 else "semantic"

    def search(self, query, k=3):
        self.last_choice = self.route(query)
        retriever = self.bm25 if self.last_choice == "bm25" else self.semantic
        return retriever.search(query, k=k)
