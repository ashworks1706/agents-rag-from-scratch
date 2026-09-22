"""Scoring for the retrieval race.

A question is a hit when the gold answer phrase appears in one of the retrieved
chunks (whitespace/case-normalised), which is fair across different chunking.
"""

import json
import re


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def load_gold(path):
    return json.load(open(path))["questions"]


def score(gold, retrieve_fn, generate_fn=None, on_progress=None):
    """retrieve_fn(question) -> list of top-k chunk texts.
    generate_fn(question, chunks) -> answer text, or None to skip answer scoring.
    on_progress(i, n, row) is called after each question, for live UI.
    Returns aggregate metrics plus a per-question 'rows' list.
    """
    n = len(gold)
    hits = 0
    rr_sum = 0.0
    answer_hits = 0
    rows = []
    for i, item in enumerate(gold, start=1):
        question = item["question"]
        ans = norm(item["answer"])
        top = retrieve_fn(question)
        rank = next((i for i, c in enumerate(top, start=1) if ans in norm(c)), None)
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
        answer_hit = None
        if generate_fn is not None:
            answer_hit = ans in norm(generate_fn(question, top))
            if answer_hit:
                answer_hits += 1
        row = {
            "question": question,
            "answer": item["answer"],
            "found": rank is not None,
            "rank": rank,
            "answer_hit": answer_hit,
        }
        rows.append(row)
        if on_progress is not None:
            on_progress(i, n, row)
    return {
        "n": n,
        "hits": hits,
        "recall": hits / n,
        "mrr": rr_sum / n,
        "answer_rate": (answer_hits / n) if generate_fn is not None else None,
        "rows": rows,
    }
