"""Router search: pick a retriever based on the query.

Different queries suit different methods. This router sends short, keyword-like
queries to BM25 and longer, natural-language queries to semantic search. Swap the
rule for an LLM classifier if you want.

pip install sentence-transformers rank-bm25 numpy
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searching.semantic_topk import SemanticSearch
from searching.bm25_search import BM25Search


class RouterSearch:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.semantic = SemanticSearch(self.chunks)
        self.bm25 = BM25Search(self.chunks)

    def route(self, query):
        # Short queries tend to be keyword lookups; longer ones tend to be semantic.
        return "bm25" if len(query.split()) <= 3 else "semantic"

    def search(self, query, k=3):
        choice = self.route(query)
        retriever = self.bm25 if choice == "bm25" else self.semantic
        return choice, retriever.search(query, k=k)


if __name__ == "__main__":
    chunks = ["Membership costs 15 dollars per semester.",
              "Workshops run every Tuesday from 6 to 8 PM.",
              "Officer elections are held once per year."]
    r = RouterSearch(chunks)
    for query in ["dues", "when can I attend a workshop this week"]:
        choice, results = r.search(query, k=1)
        print(f"[{choice}] {query!r} ->", results[0][0])
