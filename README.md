# Modern RAG in Practice

Build a RAG assistant, see how it works, and race for the best score, all in one
small web app.

## Run it

```bash
cd workshop
pip install -r requirements.txt
streamlit run app.py
```

That's it. The app opens in your browser with four tabs:

- **Ask** — ask a question, get an answer with its sources, scores, and timing.
- **Race** — score your setup on the question set and try to beat others.
- **Index** — see the chunks the current document was split into.
- **Learn** — every method in the library, each linked to the section of the
  deep-dive notebook that explains it.

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

Use the **sidebar**: pick a splitter and a search method, set chunk size /
overlap / top_k, and flip the reranker. The app re-indexes live — no code
editing, no restart. Hover a **?** icon for what each setting means, and open
the **Learn** tab for every method with a link into the deep-dive notebook.

The choices come from these folders (one file per method):

```
splitting/    how to cut the document into chunks
embedding/    how to turn text into vectors
indexing/     how to store and search vectors
searching/    ready-made retrievers (semantic, bm25, hybrid, rrf, ...)
reranking/    reorder results for accuracy
```

Prefer code? **`pipeline.py`** holds the defaults the sidebar starts from; edit
it and the app picks it up on reload.

## The race

Pick a strategy in the sidebar, open the **Race** tab, and click **Run
benchmark**. Use **Quick test (20)** while tuning and **Full (120)** for the
score you report. It scores questions over a hard SQuAD-based corpus:

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
