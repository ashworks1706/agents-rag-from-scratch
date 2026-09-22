# Retrieval race

A short competition: everyone scores the same questions against the same
document and tries to get the highest retrieval score by choosing a better
splitting and searching strategy.

Run it from the **Race** tab in the app (`streamlit run app.py`). Set the
pipeline in the sidebar, click **Run benchmark**, and read the score.

- `gold.json` — 27 questions over the sample handbook, each with a short gold
  answer phrase.
- `leaderboard.md` — paste your best result here.

## Rules

- The document and questions are fixed and the same for everyone.
- `k` is fixed for the room; you do not change it.
- You tune the splitter, chunk size and overlap, the search method, and whether
  you rerank.
- A question is a hit when the gold answer phrase appears in one of your top-k
  chunks, so different chunking is scored fairly.

## Metrics

- **Recall@k** — the headline: fraction of questions whose answer was in your
  top-k chunks.
- **MRR** — the tiebreaker: rewards ranking the right chunk first.
- **Answer@k** — optional, needs a Gemini key: the LLM answers from your top-k
  chunks and it counts if the gold phrase is in the answer.
