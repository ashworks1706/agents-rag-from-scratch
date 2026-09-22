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

The questions and corpus come from **SQuAD 1.1** (Stanford Question Answering
Dataset): 108 questions over ~150 paragraphs from six Wikipedia articles (Nikola
Tesla, oxygen, the Amazon rainforest, the Apollo program, Super Bowl 50, and
steam engines). SQuAD answers are exact text spans, which is why the substring
check works.

- `corpus.txt` — the paragraphs (the document you retrieve from).
- `gold.json` — the questions and their answers.
- `leaderboard.md` — your results.

Source: Rajpurkar et al., 2016, *SQuAD: 100,000+ Questions for Machine
Comprehension of Text*. This subset is derived from the SQuAD dev set, which is
released under CC BY-SA 4.0 and kept under the same license here.
