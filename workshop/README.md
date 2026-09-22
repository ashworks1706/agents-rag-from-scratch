# Modern RAG in Practice

Build a grounded AI assistant, see how retrieval works, race to the best score,
and deploy it, all from one small Streamlit app.

## The app

Two tabs, both driven by the method files in the stage folders:

- **Ask** — type a question and get a grounded answer with its sources, the
  similarity score for each chunk, and the retrieval and generation latency.
- **Race** — score the current pipeline on the gold question set. Edit
  `pipeline.py` to try a different strategy, then run again. You get Recall@k,
  MRR, and (with a key) Answer@k, plus a line to paste into the shared
  leaderboard.

You choose methods by editing **`pipeline.py`** and saving; the app reruns.

## Run it locally

You need Python 3.12 (3.10 or newer works). A fresh virtual environment is recommended.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here    # free key: https://aistudio.google.com/app/apikey
streamlit run app.py
```

Without a key, retrieval and scores still work and answers come back as a stub.
On first run Streamlit asks for an email; press Enter to skip it. A `torchvision`
traceback in the terminal is harmless; `streamlit run app.py --server.fileWatcherType none` silences it.

## Deploy (browser only)

Use the repo as a template (or fork it), go to https://share.streamlit.io,
create a new app, set the main file to `workshop/app.py`, add your key under
Advanced settings as a secret:

```toml
GEMINI_API_KEY = "your_key_here"
```

Then deploy. The first build installs PyTorch and downloads the embedding model,
so a few minutes is normal.

## How it's built

The pipeline is split into one folder per stage, one file per method. Each file
is a brief comment explaining the method, then the code:

```
splitting/    character, recursive, token_based, markdown_header, html_header, code_language, latex, sentence_nltk, spacy_nlp
embedding/    dense, sparse_bm25, hybrid
indexing/     numpy_flat, faiss_index, hnsw_index, chroma_index, lsh_index
searching/    semantic_topk, bm25_search, hybrid_search, query_fusion, reciprocal_rank_fusion, ensemble, router
reranking/    cross_encoder
pipeline.py   the pipeline you edit (imports one method per stage)
utils/        load_pdf and benchmark scoring
rag.py        turns retrieved chunks into a Gemini answer
app.py        the Streamlit app (Ask + Race) that runs pipeline.py
```

`pipeline.py` is the file you edit. It imports one method from each folder; swap
an import to change a method, tweak the numbers, or flip `USE_RERANKER`, then
save and the app reruns:

```python
from splitting.recursive import split
from searching.semantic_topk import SemanticSearch as Search
from reranking.cross_encoder import Reranker

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 3
USE_RERANKER = False
```

To add your own method, drop a new file in the right folder following the same
interface, then point `pipeline.py` at it.

The interfaces:

- splitting: `split(text, ...) -> list[str]`
- embedding: `DenseEmbedder().embed(texts) -> vectors` (normalised)
- indexing: `Index(vectors).query(query_vector, k) -> [(chunk_index, score)]`
- searching: `Search(chunks).search(query, k) -> [(chunk, score)]`
- reranking: `Reranker().rerank(query, chunks, k) -> [(chunk, score)]`

Some methods need extra packages (FAISS, HNSW, Chroma, spaCy) beyond the app's
requirements; see `requirements-modular.txt`.

## The race

`benchmark/gold.json` holds 27 questions over the sample handbook. The Race tab
scores your current pipeline: a question is a hit when a top-k chunk contains the
gold answer phrase, so different chunking is compared fairly. Recall@k is the
headline, MRR the tiebreaker, and Answer@k (with a key) the end-to-end score.
Keep k fixed for the room and tune everything else. Paste your best line into
`benchmark/leaderboard.md`.

## Configuration

The embedding model is `all-MiniLM-L6-v2` (small, CPU-friendly). The LLM is
Gemini (`gemini-3.6-flash`) through `google-genai`; if the model name changes,
update `GEMINI_MODEL_NAME` in `rag.py`. Defaults are 500-character chunks with
100 overlap and top-3 retrieval, all adjustable in the sidebar.
