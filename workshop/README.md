# Modern RAG in Practice

This is the workshop. In about 90 minutes you build a grounded AI assistant, see how retrieval actually works, and deploy it to a public URL. It is one continuous pipeline with no framework in the way. A document is loaded and split into chunks, each chunk is embedded, a question is matched against them with cosine similarity, and the top matches are passed to an LLM that answers using only that context. Retrieval, scoring, and latency are all visible along the way.

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

The embedding model is `all-MiniLM-L6-v2`, chosen because it is small and runs on CPU. The LLM is Gemini (`gemini-3.6-flash`) through `google-genai`, using your own key; if the model name changes, update `GEMINI_MODEL_NAME` in `rag.py`. Chunking defaults to 500-character chunks with 100 characters of overlap and returns the top 3 matches, and you can adjust all of these from the Streamlit sidebar.
