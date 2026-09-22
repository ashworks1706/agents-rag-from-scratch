# Modern RAG in Practice

This is the workshop. In about 90 minutes you build a grounded AI assistant, see how retrieval actually works, and deploy it to a public URL. It is built with LangChain and a FAISS vector store. A document is loaded and split into chunks, each chunk is embedded, a question is matched against them by cosine similarity in FAISS, and the top matches are passed to an LLM that answers using only that context. Retrieval, scoring, and latency are all visible along the way.

## Files

`rag.py` is the pipeline, and it is the same code you build step by step in the notebook. `app.py` is the Streamlit interface, including the panel that shows how each answer was produced. `slides.pptx` is the deck for the session, with speaker notes. `sample_document.pdf` is the default document, a short fictional handbook, so everyone starts from the same place. `requirements.txt`, `.env.example`, and `.streamlit/secrets.toml.example` cover the dependencies and the API key. The notebook itself is one level up, at `../tutorial.ipynb`.

## The observability panel

Every answer comes with a panel showing the chunks it retrieved, the similarity score for each one, and how long retrieval, generation, and the whole request took. The point is to make retrieval something you can inspect and debug rather than guess at.

## Running it locally

You need Python 3.12 (3.10 or newer works) installed and on your PATH. A fresh virtual environment is recommended.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here    # free key: https://aistudio.google.com/app/apikey
streamlit run app.py
```

If you do not set a key, retrieval and the metrics still work and the answer comes back as a labeled stub.

On first run Streamlit asks for an email address; press Enter to skip it. If you see a `torchvision` traceback in the terminal, it is harmless and does not affect the app; `streamlit run app.py --server.fileWatcherType none` silences it.

## Deploying

Deployment is all in the browser. Use the repo as a template (or fork it) to get your own copy, then go to https://share.streamlit.io, create a new app, and point it at your repo with `workshop/app.py` as the main file. Add your Gemini key under Advanced settings as a secret:

```toml
GEMINI_API_KEY = "your_key_here"
```

Then deploy. The first build takes a few minutes because it installs PyTorch and downloads the embedding model, which is normal.

## If you fall behind

Each stage of the pipeline is a labeled step in the notebook (Steps 1 through 6), so you can jump to the next one instead of losing the rest of the session.

## Configuration

The pipeline uses LangChain: `PyMuPDFLoader` and `RecursiveCharacterTextSplitter` for loading and chunking, `langchain-huggingface` with the `all-MiniLM-L6-v2` embedding model (small, CPU-friendly), and a FAISS vector store for retrieval. The LLM is Gemini (`gemini-3.6-flash`) through `langchain-google-genai`, using your own key; if the model name changes, update `GEMINI_MODEL_NAME` in `rag.py`. Chunking defaults to 500-character chunks with 100 characters of overlap and returns the top 3 matches, and you can adjust all of these from the Streamlit sidebar.

## Modular pipeline (experiment here)

Alongside the deployable app, the workshop is broken into one folder per RAG stage, with one file per method, so you can read, run, or swap any single piece:

```
workshop/
  splitting/    character, recursive, token_based, markdown_header, html_header, code_language, latex, sentence_nltk, spacy_nlp
  embedding/    dense, sparse_bm25, hybrid
  indexing/     numpy_flat, faiss_index, hnsw_index, chroma_index, lsh_index
  searching/    semantic_topk, bm25_search, hybrid_search, query_fusion, reciprocal_rank_fusion, ensemble, router
  reranking/    cross_encoder
  utils/        shared helpers (load_pdf, printing, name dispatch)
  main.py       the pipeline, one line per stage
```

`main.py` is deliberately just the five steps in order; the plumbing lives in `utils/`, so the core files stay easy to read.

Every file runs on its own, so you can see exactly what one method does:

```bash
pip install -r requirements-modular.txt
python splitting/recursive.py
python indexing/faiss_index.py
python searching/reciprocal_rank_fusion.py
```

`main.py` composes one method per stage (splitting, embedding, indexing, searching, reranking). To try a different method, change one import at the top of `main.py` and run it:

```bash
python main.py "How much does membership cost?"
```

The pieces share simple conventions so they fit together:

- splitting: `split(text, ...) -> list[str]`
- embedding: `DenseEmbedder().embed(texts) -> vectors` (normalised)
- indexing: `Index(vectors).query(query_vector, k) -> [(chunk_index, score)]`
- searching: `Search(chunks).search(query, k) -> [(chunk, score)]`
- reranking: `Reranker().rerank(query, chunks, k) -> [(chunk, score)]`

Notes: the deployed Streamlit app (`app.py`, `rag.py`) is separate and stays light, needing only `requirements.txt`. Some methods fetch a resource on first use: `token_based.py` downloads the tiktoken vocab, `sentence_nltk.py` downloads the NLTK sentence model, and the embedding and cross-encoder methods download their model from Hugging Face.

## Retrieval race

`benchmark/` holds a small competition: fixed questions and document, and you race to the highest retrieval score by choosing a better splitter, embedding, and search method. See `benchmark/README.md`.
