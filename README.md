# agents-rag-from-scratch

Two things live here:

1. **The workshop** — a small web app that answers questions about a document,
   shows its sources, and lets you compete for the best retrieval score. See
   [`workshop/`](workshop/).
2. **The deep dive** — [`tutorial.ipynb`](tutorial.ipynb), a long reference
   notebook on agents, retrieval, evaluation, and more. Open it in Google Colab
   and read at your own pace.

## Start the workshop

```bash
cd workshop
pip install -r requirements.txt
streamlit run app.py
```

A free Gemini key (below) turns on the AI answers. Without it, everything else
still works.

See [`workshop/README.md`](workshop/README.md) for the full guide.

## License

See [`LICENSE`](LICENSE).
