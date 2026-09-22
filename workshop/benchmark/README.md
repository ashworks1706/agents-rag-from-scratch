# Retrieval race

A short competition: everyone runs the same questions against the same document
and tries to get the highest retrieval score by choosing a better splitting,
embedding, and searching strategy.

## The rules

- The document (`../sample_document.pdf`) and the questions (`gold.json`) are
  fixed and the same for everyone.
- `k` is fixed for the whole room (default `k=3`). You do **not** change `k`.
- You tune everything else: the splitter and its chunk size/overlap, the search
  method, and whether you rerank.
- A question counts as a hit when the gold answer phrase appears in one of your
  top-k chunks, so different chunking is scored fairly.

## Metrics

- **Recall@k** is the headline: the fraction of questions whose answer was in
  your top-k chunks.
- **MRR** is the tiebreaker: it rewards ranking the right chunk first.
- **Answer@k** (optional, needs a Gemini key) is the end-to-end score: the LLM
  answers using your top-k chunks, and it counts if the gold phrase is in the
  answer.

## Running it

```bash
pip install -r ../requirements-modular.txt
python benchmark/evaluate.py --search semantic --splitter recursive --chunk-size 500 --overlap 100
```

Things to try, and roughly what they change:

```bash
# smaller vs larger chunks
python benchmark/evaluate.py --search semantic --chunk-size 200 --overlap 40
python benchmark/evaluate.py --search semantic --chunk-size 600 --overlap 120

# different search methods
python benchmark/evaluate.py --search bm25
python benchmark/evaluate.py --search hybrid
python benchmark/evaluate.py --search rrf

# add a cross-encoder reranker on top of a first-stage retriever
python benchmark/evaluate.py --search semantic --rerank

# end-to-end answer accuracy with the LLM
export GEMINI_API_KEY=your_key_here
python benchmark/evaluate.py --search hybrid --rerank --llm --name "Team Bass"
```

Each run prints a `LEADERBOARD` line. Put your best one in `leaderboard.md`.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--splitter` | recursive | splitting method (a file in `splitting/`) |
| `--chunk-size` | 500 | characters per chunk (recursive/character/token_based/nltk/spacy) |
| `--overlap` | 100 | overlap between chunks |
| `--search` | semantic | semantic, bm25, hybrid, query_fusion, rrf, ensemble, router |
| `--k` | 3 | chunks scored (fixed for the race) |
| `--rerank` | off | apply the cross-encoder reranker |
| `--pool` | 20 | candidates retrieved before reranking |
| `--llm` | off | also score end-to-end answer accuracy |
| `--name` | | label for your leaderboard line |
