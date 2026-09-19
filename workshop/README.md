# Modern RAG in Practice — Workshop

**Build a grounded AI assistant, inspect how retrieval works, and deploy it as a live application — in 90 minutes.**

One continuous pipeline, no framework:

```
PDF  ->  chunks  ->  embeddings  ->  cosine similarity  ->  top-k  ->  LLM answer  ->  sources + metrics  ->  Streamlit
```

No LangChain, no vector database, no Docker. Every line is readable.

## What's here

| File | Purpose |
|------|---------|
| `../tutorial.ipynb` | The Colab notebook — workshop section on top, full deep-dive reference below |
| `rag.py` | The reusable pipeline (the code you build in the notebook) |
| `app.py` | Streamlit interface with the observability panel |
| `requirements.txt` | Dependencies |
| `.env.example` | Environment-variable template |
| `.streamlit/secrets.toml.example` | Streamlit Cloud secrets template |
| `sample_document.pdf` | Sample PDF used as the default document |

## The observability panel

Every answer shows a **"How was this answer generated?"** panel:

- the retrieved chunks (your sources),
- the similarity score for each,
- retrieval time, generation time, and total request time.

That is what turns RAG from magic into something you can reason about.

## Run it locally

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here    # free: https://aistudio.google.com/app/apikey
streamlit run app.py
```

No key? Retrieval, scores, and latency still work; the answer becomes a labeled stub.

## Deploy (browser only)

1. **Use this template** (or fork) to get your own copy of the repo.
2. Go to https://share.streamlit.io → **New app** → select your repo.
3. **Main file path:** `workshop/app.py`.
4. **Advanced settings → Secrets:**
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
5. **Deploy.** The first build installs PyTorch and downloads the embedding model — a few minutes is normal.

## Checkpoints

If you fall behind, each stage of the pipeline is a labeled step in the notebook
(Steps 1–6). Jump to the next step rather than losing the rest of the session.

## Configuration notes

- **Embedding model:** `all-MiniLM-L6-v2` (small, CPU-friendly).
- **LLM:** Gemini (`gemini-2.0-flash`) via `google-genai`, bring-your-own-key.
  Change `GEMINI_MODEL_NAME` in `rag.py` if the model name changes.
- **Chunking:** 500-char chunks, 100-char overlap, top-3 retrieval — all adjustable
  in the Streamlit sidebar.
