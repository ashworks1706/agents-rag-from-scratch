"""Rebuild the benchmark from SQuAD.

Pulls SQuAD 1.1 dev (Rajpurkar et al., 2016, CC BY-SA 4.0), builds a corpus of
Wikipedia paragraphs and a HARD question set (questions that plain BM25 already
ranks first are dropped), and writes corpus.txt + gold.json.

Primary source: the Hugging Face datasets hub. Falls back to the SQuAD GitHub
mirror when the hub is unavailable.

    python benchmark/build_dataset.py
"""

import json
import os
import re
import random

from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = os.path.dirname(os.path.abspath(__file__))
N_ARTICLES = 20
N_QUESTIONS = 120

norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()


def load_squad_articles():
    try:
        from datasets import load_dataset
        ds = load_dataset("squad", split="validation")
        by_title = {}
        for row in ds:
            by_title.setdefault(row["title"], []).append({
                "context": row["context"],
                "question": row["question"],
                "answer": row["answers"]["text"][0],
            })
        print("loaded SQuAD from the Hugging Face hub")
        return [{"rows": rows} for rows in by_title.values()]
    except Exception as exc:
        print(f"Hugging Face datasets unavailable ({exc.__class__.__name__}); using GitHub mirror")

    import urllib.request
    url = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
    data = json.loads(urllib.request.urlopen(url, timeout=60).read())
    articles = []
    for a in data["data"]:
        rows = []
        for p in a["paragraphs"]:
            for qa in p["qas"]:
                rows.append({"context": p["context"], "question": qa["question"], "answer": qa["answers"][0]["text"]})
        articles.append({"rows": rows})
    return articles


def main():
    articles = load_squad_articles()[:N_ARTICLES]

    paragraphs, seen, candidates = [], set(), []
    for art in articles:
        for r in art["rows"]:
            ctx = r["context"].strip()
            if ctx not in seen:
                seen.add(ctx)
                paragraphs.append(ctx)
            ans = r["answer"].strip()
            if 0 < len(ans) <= 40:
                candidates.append({"question": r["question"].strip(), "answer": ans})

    corpus = "\n\n".join(paragraphs)
    ncorpus = norm(corpus)
    candidates = [x for x in candidates if norm(x["answer"]) in ncorpus]

    seen_q, unique = set(), []
    for x in candidates:
        k = norm(x["question"])
        if k not in seen_q:
            seen_q.add(k)
            unique.append(x)

    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_text(corpus)
    bm25 = BM25Okapi([c.lower().split() for c in chunks])

    def first_rank(question, answer):
        scores = bm25.get_scores(question.lower().split())
        order = sorted(range(len(chunks)), key=lambda i: -scores[i])[:20]
        for pos, i in enumerate(order, start=1):
            if norm(answer) in norm(chunks[i]):
                return pos
        return None

    hard = []
    for x in unique:
        rank = first_rank(x["question"], x["answer"])
        if rank is not None and rank >= 2:   # drop gimmes BM25 nails at rank 1
            hard.append(x)
    random.Random(0).shuffle(hard)
    hard = hard[:N_QUESTIONS]

    open(os.path.join(HERE, "corpus.txt"), "w", encoding="utf-8").write(corpus)
    json.dump({
        "source": "SQuAD 1.1 dev (Rajpurkar et al., 2016), CC BY-SA 4.0 — hard subset (gimmes removed)",
        "corpus_file": "corpus.txt",
        "questions": hard,
    }, open(os.path.join(HERE, "gold.json"), "w", encoding="utf-8"), indent=2)

    print(f"paragraphs={len(paragraphs)} chunks={len(chunks)} questions={len(hard)}")


if __name__ == "__main__":
    main()
