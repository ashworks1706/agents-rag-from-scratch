# The race

Everyone scores the same questions on the same corpus. Best retrieval score wins.

## How to play

1. Edit `pipeline.py` to pick your strategy (splitter, search method, chunk
   size, reranking).
2. Open the **Race** tab in the app and click **Run benchmark**.
3. Paste your best line into `leaderboard.md`.

Keep `TOP_K` the same as everyone else. Change anything else.

## The scores

- **Recall@k** — was the answer in your top results? This is the one to beat.
- **MRR** — how high did you rank the right chunk? (tiebreaker)
- **Answer@k** — did the AI answer correctly? (needs a Gemini key)

A question counts if the answer text appears in one of your top chunks, so
different chunk sizes are compared fairly.

## The data

From **SQuAD 1.1** (Stanford Question Answering Dataset): 120 questions over ~900
Wikipedia paragraphs (about 1,900 chunks). It is a **hard subset** — questions
that plain keyword search (BM25) already ranks first are removed, so you have to
retrieve well to score. As a baseline, BM25 alone reaches only Recall@3 ≈ 0.41.

- `corpus.txt` — the paragraphs you retrieve from.
- `gold.json` — the questions and answers.
- `leaderboard.md` — your results.

## Regenerate or find the best pipeline

```bash
python benchmark/build_dataset.py   # rebuild corpus.txt + gold.json from SQuAD (Hugging Face, or GitHub fallback)
python benchmark/sweep.py           # score every splitter x search x rerank combo, print the winner
```

`sweep.py` needs the models (Hugging Face download), so run it where there is
internet, e.g. Codespaces.

Source: Rajpurkar et al., 2016, *SQuAD: 100,000+ Questions for Machine
Comprehension of Text*. SQuAD is released under CC BY-SA 4.0; this subset keeps
the same license.
