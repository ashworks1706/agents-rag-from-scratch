"""Streamlit app for the Modern RAG workshop.

The pipeline is defined in pipeline.py; edit that file to change methods. This
app just runs it, in two tabs:
  Ask   - ask a question, see the grounded answer, sources, scores, and latency.
  Race  - score the current pipeline on the gold question set (Recall@k, MRR).
"""

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from utils import load_pdf
from utils.benchmark import load_gold, score
import rag
import pipeline

importlib.reload(pipeline)  # pick up edits to pipeline.py on each rerun

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOC = os.path.join(APP_DIR, "sample_document.pdf")
GOLD = os.path.join(APP_DIR, "benchmark", "gold.json")

st.set_page_config(page_title="Modern RAG in Practice", page_icon="🔎", layout="centered")

try:
    if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass


def pipeline_sig():
    """Identity of the current pipeline, so caches rebuild when you edit pipeline.py."""
    return (pipeline.split.__module__, pipeline.Search.__module__, pipeline.Search.__name__,
            pipeline.CHUNK_SIZE, pipeline.OVERLAP)


@st.cache_resource(show_spinner="Indexing the document...")
def build_cached(doc_path, sig):
    return pipeline.build(load_pdf(doc_path))


@st.cache_resource(show_spinner="Loading reranker...")
def get_reranker(name):
    return pipeline.Reranker()


def retrieve_scored(searcher, query):
    n = pipeline.TOP_K * 4 if pipeline.USE_RERANKER else pipeline.TOP_K
    results = searcher.search(query, k=n)
    if isinstance(results, tuple):
        results = results[1]
    if pipeline.USE_RERANKER:
        candidates = [c for c, _ in results]
        results = get_reranker(pipeline.Reranker.__name__).rerank(query, candidates, k=pipeline.TOP_K)
    return results[:pipeline.TOP_K]


st.title("🔎 Modern RAG in Practice")

with st.sidebar:
    st.header("Pipeline")
    st.write("Edit **`pipeline.py`** to change the splitter, search method, or reranker, then save.")
    st.code(
        f"split    = {pipeline.split.__module__.split('.')[-1]}\n"
        f"search   = {pipeline.Search.__module__.split('.')[-1]}\n"
        f"chunk    = {pipeline.CHUNK_SIZE} / {pipeline.OVERLAP}\n"
        f"top_k    = {pipeline.TOP_K}\n"
        f"rerank   = {pipeline.USE_RERANKER}",
        language="text",
    )
    st.divider()
    uploaded = st.file_uploader("Document (optional PDF)", type=["pdf"])
    if os.environ.get("GEMINI_API_KEY"):
        st.success("Gemini key detected.")
    else:
        st.warning("No Gemini key: retrieval and scores work; answers are stubs.")

doc_path = DEFAULT_DOC
if uploaded is not None:
    doc_path = os.path.join(APP_DIR, f"_uploaded_{uploaded.name}")
    with open(doc_path, "wb") as fh:
        fh.write(uploaded.getvalue())

chunks, searcher = build_cached(doc_path, pipeline_sig())
st.caption(f"{len(chunks)} chunks indexed.")

tab_ask, tab_race = st.tabs(["Ask", "Race"])

with tab_ask:
    question = st.text_input("Your question", placeholder="e.g. How much does membership cost?")
    if st.button("Ask", type="primary") and question.strip():
        t0 = time.perf_counter()
        scored = retrieve_scored(searcher, question)
        t1 = time.perf_counter()
        rc = [rag.RetrievedChunk(text=c, score=s, index=i) for i, (c, s) in enumerate(scored)]
        answer, used_llm, warning = rag.generate_answer(question, rc)
        t2 = time.perf_counter()

        st.subheader("Answer")
        st.write(answer)
        if warning:
            st.caption(f"⚠️ {warning}")

        with st.expander("How was this answer generated?", expanded=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Retrieval", f"{(t1 - t0) * 1000:.0f} ms")
            c2.metric("Generation", f"{(t2 - t1) * 1000:.0f} ms")
            c3.metric("Total", f"{(t2 - t0) * 1000:.0f} ms")
            st.caption(f"LLM used: {'yes' if used_llm else 'no (stub answer)'}")
            for rank, (chunk, s) in enumerate(scored, start=1):
                st.markdown(f"**Source {rank}** · similarity `{s:.3f}`")
                st.progress(max(0.0, min(1.0, float(s))))
                st.write(chunk)
                st.divider()

with tab_race:
    st.write("Score the current pipeline on the gold question set. "
             "Edit `pipeline.py` to try a different strategy, then run again. "
             "Keep **TOP_K** the same as the rest of the room.")
    if st.button("Run benchmark", type="primary"):
        gold = load_gold(GOLD)
        with st.spinner(f"Scoring {len(gold)} questions..."):
            def retrieve_fn(q):
                return [c for c, _ in retrieve_scored(searcher, q)]

            generate_fn = None
            if os.environ.get("GEMINI_API_KEY"):
                def generate_fn(q, top):
                    rc = [rag.RetrievedChunk(text=c, score=0.0, index=i) for i, c in enumerate(top)]
                    return rag.generate_answer(q, rc)[0]

            result = score(gold, retrieve_fn, generate_fn)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Recall@{pipeline.TOP_K}", f"{result['recall']:.3f}", f"{result['hits']}/{result['n']}")
        c2.metric("MRR", f"{result['mrr']:.3f}")
        if result["answer_rate"] is not None:
            c3.metric(f"Answer@{pipeline.TOP_K}", f"{result['answer_rate']:.3f}")
        else:
            c3.caption("Answer@k needs a Gemini key")

        label = f"{pipeline.split.__module__.split('.')[-1]}/{pipeline.Search.__module__.split('.')[-1]}"
        if pipeline.USE_RERANKER:
            label += "+rerank"
        line = f"{label:<28} Recall@{pipeline.TOP_K}={result['recall']:.3f}  MRR={result['mrr']:.3f}"
        if result["answer_rate"] is not None:
            line += f"  Answer@{pipeline.TOP_K}={result['answer_rate']:.3f}"
        st.caption("Copy your best line into benchmark/leaderboard.md:")
        st.code(line, language="text")
