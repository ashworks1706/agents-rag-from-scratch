"""Find the golden pipeline: score every method combination on the benchmark.

Runs each splitter x search-method x rerank over the benchmark corpus and prints
a leaderboard sorted by Recall@k. This needs the embedding + reranker models
(Hugging Face download), so run it where there is internet, e.g. Codespaces:

    python benchmark/sweep.py            # TOP_K = 3
    python benchmark/sweep.py 5          # TOP_K = 5
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSHOP = os.path.dirname(HERE)
sys.path.insert(0, WORKSHOP)

from utils.benchmark import load_gold, score

CORPUS = os.path.join(HERE, "corpus.txt")
GOLD = os.path.join(HERE, "gold.json")

SPLITTERS = ["recursive", "character"]
SEARCHES = ["semantic_topk", "bm25_search", "hybrid_search", "query_fusion",
            "reciprocal_rank_fusion", "ensemble"]
CHUNK_SIZE, OVERLAP = 500, 100

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from reranking.cross_encoder import Reranker
        _reranker = Reranker()
    return _reranker


def make_searcher(search, chunks):
    mod = __import__(f"searching.{search}", fromlist=["*"])
    cls = next(getattr(mod, n) for n in dir(mod)
               if n[0].isupper() and hasattr(getattr(mod, n), "search"))
    return cls(chunks)


def retrieve(searcher, query, k, rerank):
    n = k * 4 if rerank else k
    results = searcher.search(query, k=n)
    if isinstance(results, tuple):
        results = results[1]
    if rerank:
        results = get_reranker().rerank(query, [c for c, _ in results], k=k)
    return [c for c, _ in results[:k]]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    text = open(CORPUS, encoding="utf-8").read()
    gold = load_gold(GOLD)

    from splitting import recursive, character
    splitters = {"recursive": recursive, "character": character}

    board = []
    for sp_name in SPLITTERS:
        chunks = splitters[sp_name].split(text, CHUNK_SIZE, OVERLAP)
        for se_name in SEARCHES:
            searcher = make_searcher(se_name, chunks)
            for rerank in (False, True):
                res = score(gold, lambda q, s=searcher, rr=rerank: retrieve(s, q, k, rr))
                name = f"{sp_name}/{se_name}" + ("+rerank" if rerank else "")
                board.append((res["recall"], res["mrr"], name))
                print(f"  {name:<42} Recall@{k}={res['recall']:.3f} MRR={res['mrr']:.3f}")

    print("\n=== leaderboard (by Recall@%d, then MRR) ===" % k)
    for recall, mrr, name in sorted(board, reverse=True):
        print(f"  {name:<42} Recall@{k}={recall:.3f} MRR={mrr:.3f}")
    print(f"\nGOLDEN: {sorted(board, reverse=True)[0][2]}")


if __name__ == "__main__":
    main()
