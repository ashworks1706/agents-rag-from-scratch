"""Benchmark harness for the workshop retrieval race.

Scores a retrieval pipeline against a gold question/answer set (gold.json).
A question is a "hit" when any retrieved chunk contains the gold answer phrase,
which keeps scoring fair even though attendees chunk the document differently.

Metrics:
  Recall@k   fraction of questions whose gold answer appears in the top-k chunks
  MRR        mean reciprocal rank of the first chunk that contains the answer
  Answer@k   (with --llm) fraction of questions the LLM answers with the gold phrase

Examples:
  python benchmark/evaluate.py --search semantic --splitter recursive --chunk-size 500 --overlap 100 --k 5
  python benchmark/evaluate.py --search hybrid --splitter recursive --rerank --k 5 --name "Ada"
  python benchmark/evaluate.py --search rrf --llm --name "Team Bass"
"""

import argparse
import json
import os
import re
import sys

WORKSHOP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSHOP)

DOC = os.path.join(WORKSHOP, "sample_document.pdf")
GOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold.json")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def load_text(path):
    import pymupdf
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def get_chunks(splitter, text, chunk_size, overlap):
    """Dispatch to a splitting module. Size/overlap apply where the method supports them."""
    if splitter in ("recursive", "character", "token_based"):
        mod = __import__(f"splitting.{splitter}", fromlist=["split"])
        return mod.split(text, chunk_size=chunk_size, overlap=overlap)
    if splitter in ("sentence_nltk", "spacy_nlp"):
        mod = __import__(f"splitting.{splitter}", fromlist=["split"])
        return mod.split(text, chunk_size=chunk_size)
    # header/code/latex splitters expect structured input; fall back to plain split
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splitter", default="recursive")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=100)
    ap.add_argument("--search", default="semantic")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=20, help="candidates to retrieve before reranking")
    ap.add_argument("--rerank", action="store_true", help="apply cross-encoder reranking")
    ap.add_argument("--llm", action="store_true", help="also score end-to-end answer accuracy (needs GEMINI_API_KEY)")
    ap.add_argument("--name", default="", help="label for the leaderboard line")
    args = ap.parse_args()

    gold = json.load(open(GOLD))["questions"]
    text = load_text(DOC)
    chunks = get_chunks(args.splitter, text, args.chunk_size, args.overlap)
    searcher = build_search(args.search, chunks)

    reranker = None
    if args.rerank:
        from reranking.cross_encoder import Reranker
        reranker = Reranker()

    llm_ready = args.llm and os.environ.get("GEMINI_API_KEY", "").strip()

    hits = 0
    rr_sum = 0.0
    answer_hits = 0
    for item in gold:
        q, ans = item["question"], norm(item["answer"])
        if reranker:
            candidates = retrieve(searcher, q, args.pool)
            top = [c for c, _ in reranker.rerank(q, candidates, k=args.k)]
        else:
            top = retrieve(searcher, q, args.k)

        rank = next((i for i, c in enumerate(top, start=1) if ans in norm(c)), None)
        if rank:
            hits += 1
            rr_sum += 1.0 / rank

        if llm_ready:
            import rag
            rc = [rag.RetrievedChunk(text=c, score=0.0, index=i) for i, c in enumerate(top)]
            answer, _used, _warn = rag.generate_answer(q, rc)
            if ans in norm(answer):
                answer_hits += 1

    n = len(gold)
    recall = hits / n
    mrr = rr_sum / n
    label = args.name or f"{args.splitter}/{args.search}" + ("+rerank" if args.rerank else "")

    print(f"\nBenchmark: {n} questions on {os.path.basename(DOC)}")
    print(f"  splitter={args.splitter} chunk_size={args.chunk_size} overlap={args.overlap} "
          f"search={args.search} k={args.k} rerank={args.rerank}")
    print(f"  chunks indexed: {len(chunks)}")
    print(f"\n  Recall@{args.k}: {recall:.3f}   ({hits}/{n})")
    print(f"  MRR:       {mrr:.3f}")
    if args.llm:
        if llm_ready:
            print(f"  Answer@{args.k}: {answer_hits / n:.3f}   ({answer_hits}/{n})")
        else:
            print("  Answer@k: skipped (set GEMINI_API_KEY and pass --llm)")
    print(f"\nLEADERBOARD  {label:<28} Recall@{args.k}={recall:.3f}  MRR={mrr:.3f}"
          + (f"  Answer@{args.k}={answer_hits/n:.3f}" if llm_ready else ""))


if __name__ == "__main__":
    main()
