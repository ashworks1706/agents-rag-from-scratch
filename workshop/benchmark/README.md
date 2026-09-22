# The race

Everyone scores the same 27 questions on the same document. Best retrieval score
wins.

## How to play

1. Edit `pipeline.py` to pick your strategy (splitter, search method, chunk
   size, reranking).
2. Open the **Race** tab in the app and click **Run benchmark**.
3. Read your score and paste your best line into `leaderboard.md`.

Keep `TOP_K` the same as everyone else. Change anything else.

## The scores

- **Recall@k** — was the answer in your top results? This is the one to beat.
- **MRR** — how high did you rank the right chunk? (tiebreaker)
- **Answer@k** — did the AI answer correctly? (needs a Gemini key)

A question counts as correct if the answer text shows up in one of your top
chunks, so different chunk sizes are compared fairly.

## Files

- `gold.json` — the 27 questions and their answers.
- `leaderboard.md` — put your best result here.
