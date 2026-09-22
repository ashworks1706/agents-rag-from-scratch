# agents-rag-from-scratch

This repository holds two things, both built around a single notebook.

The first is a 90-minute workshop called Modern RAG in Practice. It walks through building a retrieval-augmented generation assistant from scratch and deploying it as a web app. You load a document, split it into chunks, embed those chunks, retrieve the most relevant ones from a FAISS vector store, and pass them to an LLM to get an answer that is grounded in the document and cites its sources. It is built with LangChain and FAISS, both free and running locally with no server to set up. The workshop lives at the top of `tutorial.ipynb`, and the code you deploy is in the `workshop/` folder. See `workshop/README.md` for how to run and deploy it.

The second is a deep-dive reference, which is the rest of `tutorial.ipynb` below the workshop section. It is a much broader tour: agents, prompting, tools and MCP, memory, workflows and routing, the full retrieval stack (dense, sparse, and hybrid embeddings, and re-ranking), vector stores like FAISS and Chroma, knowledge graphs, evaluation metrics, and guardrails. It is meant for reading and experimenting at your own pace rather than following along live.

## Quick start

```bash
cd workshop
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # free key: https://aistudio.google.com/app/apikey
streamlit run app.py
```

You can also open `tutorial.ipynb` in Google Colab and run the workshop section from the top. A free Gemini key turns on answer generation; without one, retrieval and the metrics still work and answers come back as a labeled stub.

## Layout

```
agents-rag-from-scratch/
├── tutorial.ipynb          workshop section on top, deep-dive reference below
├── workshop/
│   ├── rag.py              the RAG pipeline
│   ├── app.py              Streamlit interface with the observability panel
│   ├── slides.pptx         the presentation deck
│   ├── sample_document.pdf the default document
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── LICENSE
└── README.md
```

## License

See `LICENSE`.
