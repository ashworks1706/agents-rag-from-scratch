# agents-rag-from-scratch

A hands-on guide to Retrieval-Augmented Generation and agentic systems — from first principles through a deployed application.

This repository serves two audiences from one notebook:

- **🎓 Workshop (90 minutes):** *Modern RAG in Practice* — build one clean RAG pipeline and deploy it as a live web app.
- **📚 Deep dive (self-paced):** the full technical reference on agents, retrieval, evaluation, and guardrails.

## Two ways in

### 1. The 90-minute workshop — *Modern RAG in Practice*

Build a grounded assistant end to end and ship it:

```
PDF -> chunks -> embeddings -> cosine similarity -> top-k -> LLM answer -> sources + metrics -> Streamlit
```

- Start in [`tutorial.ipynb`](tutorial.ipynb) — the **workshop section is at the top**.
- All deployable code is in [`workshop/`](workshop/) (`rag.py`, `app.py`, `requirements.txt`, sample document).
- Deploy to a public URL with Streamlit Community Cloud — see [`workshop/README.md`](workshop/README.md).

No LangChain, no vector database, no Docker. Every line is readable.

### 2. The full technical deep dive

The lower section of [`tutorial.ipynb`](tutorial.ipynb) is a comprehensive reference covering:

- agents, prompting, tools, and the Model Context Protocol (MCP);
- context engineering and memory systems;
- workflows, chains, routing, and parallelization;
- the full RAG stack: loading, chunking, dense/sparse/hybrid embeddings, re-ranking;
- vector stores (FAISS, Chroma, HNSW) and knowledge graphs;
- retrieval strategies, evaluation metrics, and guardrails.

## Repository layout

```
agents-rag-from-scratch/
├── tutorial.ipynb          # workshop (top) + deep-dive reference (below)
├── workshop/
│   ├── rag.py              # the reusable RAG pipeline
│   ├── app.py              # Streamlit interface + observability panel
│   ├── requirements.txt
│   ├── .env.example
│   ├── sample_document.pdf # fictional AI Society handbook
│   └── README.md           # workshop + deployment guide
├── LICENSE
└── README.md
```

## Quick start (workshop)

```bash
cd workshop
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # free: https://aistudio.google.com/app/apikey
streamlit run app.py
```

Or open `tutorial.ipynb` in Google Colab and run the workshop section top to bottom.

## Scope

This is an educational, transparent implementation — small and explicit rather than a production framework. Use it to understand how the pieces work, then extend them.

## License

See [`LICENSE`](LICENSE).
