# Modern RAG in Practice

Build a RAG assistant, see how it works, and race for the best score, all in one
small web app.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

That's it. The app opens in your browser with two tabs:

- **Ask** — ask a question, get an answer with its sources, scores, and timing.
- **Race** — score your setup on a fixed question set and try to beat others.

Works on your laptop, in GitHub Codespaces, or deployed (see below).

## Do you need an API key?

One, and it's optional:

- **`GEMINI_API_KEY`** — free at https://aistudio.google.com/app/apikey (no card).
  It only powers the AI *answer*. Splitting, search, and all the scores work
  without it.

Set it before running:

```bash
export GEMINI_API_KEY=your_key_here
```

## Change the pipeline

Everything the app runs is in **`pipeline.py`**. Edit it, save, and the app
reloads:

```python
from splitting.recursive import split                        # the splitter
from searching.semantic_topk import SemanticSearch as Search  # the search method
from reranking.cross_encoder import Reranker

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
USE_RERANKER = False
```

Swap an import to try a different method, change a number, or flip
`USE_RERANKER`. The choices come from these folders (one file per method):

```
splitting/    how to cut the document into chunks
embedding/    how to turn text into vectors
indexing/     how to store and search vectors
searching/    ready-made retrievers (semantic, bm25, hybrid, rrf, ...)
reranking/    reorder results for accuracy
```

Open any file to see how that method works.

## The race

Pick a strategy in `pipeline.py`, open the **Race** tab, and click **Run
benchmark**. It scores 108 questions over a SQuAD-based corpus:

- **Recall@k** — did the answer show up in your top results? (main score)
- **MRR** — did it rank the right chunk high? (tiebreaker)
- **Answer@k** — did the AI answer correctly? (needs a key)

Everyone uses the same `TOP_K`. Tune the rest and paste your best line into
`benchmark/leaderboard.md`.

## Deploy it

Push to GitHub, go to https://share.streamlit.io, pick your repo, set the main
file to `workshop/app.py`, add `GEMINI_API_KEY` under Secrets, and deploy. You
get a public link. The first build takes a few minutes.

## Notes

- Model: `all-MiniLM-L6-v2` embeddings (CPU), Gemini `gemini-3.6-flash` for
  answers (change `GEMINI_MODEL_NAME` in `rag.py` if needed).
- Some methods in the folders need extra packages (FAISS, spaCy, ...); see
  `requirements-modular.txt`.
- To collect feedback and scores, set `FEEDBACK_FORM_URL` in `app.py` to your
  Google Form link; a **Submit feedback & scores** button then appears after the
  race results and in the sidebar.
