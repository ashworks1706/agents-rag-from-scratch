# agents-rag-from-scratch

This repository holds two things, both built around a single notebook.

The first is a 90-minute workshop called Modern RAG in Practice. It is a small Streamlit app that loads a document, splits it into chunks, embeds them, retrieves the most relevant ones, and passes them to an LLM for a grounded answer that cites its sources. The app has an Ask tab and a Race tab where you compete on retrieval accuracy, and the pipeline is broken into one folder per stage (splitting, embedding, indexing, searching, reranking) so any method can be read or swapped. It lives in the `workshop/` folder; see `workshop/README.md` for how to run and deploy it.

The second is a deep-dive reference, `tutorial.ipynb`: a much broader tour of agents, prompting, tools and MCP, memory, workflows and routing, the full retrieval stack (dense, sparse, and hybrid embeddings, and re-ranking), vector stores like FAISS and Chroma, knowledge graphs, evaluation metrics, and guardrails. It is meant for reading and experimenting at your own pace rather than following along live.

## Quick start

```bash
cd workshop
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # free key: https://aistudio.google.com/app/apikey
streamlit run app.py
```

Open `tutorial.ipynb` in Google Colab for the deep-dive reference. A free Gemini key turns on answer generation in the app; without one, retrieval and the scores still work and answers come back as a labeled stub.

## Layout

```
agents-rag-from-scratch/
├── tutorial.ipynb          deep-dive reference notebook
├── workshop/
│   ├── app.py              the Streamlit app (Ask + Race tabs)
│   ├── rag.py              turns retrieved chunks into a Gemini answer
│   ├── splitting/ embedding/ indexing/ searching/ reranking/   one file per method
│   ├── utils/             load_pdf, method dispatch, benchmark scoring
│   ├── benchmark/         gold questions + leaderboard for the race
│   ├── slides.pptx         the presentation deck
│   ├── sample_document.pdf the default document
│   ├── requirements.txt
│   └── README.md
├── LICENSE
└── README.md
```

## License

See `LICENSE`.
