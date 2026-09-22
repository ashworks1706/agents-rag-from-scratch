"""Streamlit app for the Modern RAG workshop.

Edit pipeline.py to change methods; this app runs it, in three tabs:
  Ask    - ask a question, see the answer, sources, and latency.
  Race   - score the pipeline on the benchmark (Recall@k / MRR).
  Index  - see the chunks the current document was split into.
"""

import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from utils import load_pdf
from utils.benchmark import load_gold, score
import rag
import pipeline

importlib.reload(pipeline)  # pick up edits to pipeline.py on each rerun

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DOC = os.path.join(APP_DIR, "sample_document.pdf")
CORPUS = os.path.join(APP_DIR, "benchmark", "corpus.txt")
GOLD = os.path.join(APP_DIR, "benchmark", "gold.json")

# Paste your Google Form link here to show a "Submit feedback & scores" button.
FEEDBACK_FORM_URL = "https://forms.gle/rYatUf94Sx1TqXFN7"

st.set_page_config(page_title="Modern RAG in Practice", layout="wide")

st.markdown(
    """
    <style>
      html, body, [class*="css"], .stApp, button, input, textarea, code, pre {
        font-family: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
      }
      .stApp { background: #000; }
      h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; }
      section[data-testid="stSidebar"] { min-width: 240px; max-width: 260px; }
      .stButton > button, .stFormSubmitButton > button {
        border-radius: 6px; border: 1px solid #2a2a2a; background: #fff; color: #000; font-weight: 600;
      }
      .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #000; color: #fff; border-color: #fff;
      }
      .stLinkButton > a { border-radius: 6px; border: 1px solid #2a2a2a; background: #0a0a0a; color: #fff; }
      .stLinkButton > a:hover { border-color: #fff; }
      [data-testid="stMetric"], [data-testid="stDataFrame"], .stCode, pre {
        border: 1px solid #1c1c1c; border-radius: 8px;
      }
      [data-testid="stMetric"] { padding: 12px 14px; }
      .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass


def pipeline_sig():
    return (pipeline.split.__module__, pipeline.Search.__module__, pipeline.Search.__name__,
            pipeline.CHUNK_SIZE, pipeline.OVERLAP)


def pipeline_diagram_html():
    """A left-to-right node diagram of the current pipeline."""
    has_key = bool(os.environ.get("GEMINI_API_KEY"))

    def node(label, value, sub="", dim=False):
        color = "#4a4a4a" if dim else "#e6e6e6"
        border = "#161616" if dim else "#2a2a2a"
        sub_html = f'<div style="font-size:11px;color:#7a7a7a">{sub}</div>' if sub else ""
        return (
            f'<div style="border:1px solid {border};border-radius:8px;padding:7px 12px;'
            f'background:#0a0a0a;text-align:center;color:{color}">'
            f'<div style="font-size:11px;color:#7a7a7a">{label}</div>'
            f'<div style="font-weight:700">{value}</div>{sub_html}</div>'
        )

    arrow = '<div style="color:#5a5a5a">&rarr;</div>'
    nodes = [
        node("input", "document"),
        node("split", pipeline.split.__module__.split(".")[-1], f"{pipeline.CHUNK_SIZE}/{pipeline.OVERLAP}"),
        node("search", pipeline.Search.__module__.split(".")[-1], f"top {pipeline.TOP_K}"),
        node("rerank", "cross_encoder" if pipeline.USE_RERANKER else "off", dim=not pipeline.USE_RERANKER),
        node("answer", "gemini" if has_key else "stub", dim=not has_key),
    ]
    return (
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
        'margin:4px 0 16px;font-family:ui-monospace,SF Mono,Menlo,monospace">'
        + arrow.join(nodes) + "</div>"
    )


def build_index(cache_key, load_text, sig, label):
    """Build (chunks, searcher) once, narrating each real stage in a live
    status box. Cached in session_state, so a rebuild only runs when the
    pipeline signature changes; cache hits return instantly with no box.
    """
    key = f"index::{cache_key}::{sig}"
    if key in st.session_state:
        return st.session_state[key]

    box = st.empty()
    with box:
        status = st.status(label, expanded=True)
    state = {"chunks": 0}

    def on_stage(name, n=None):
        if name == "split":
            state["chunks"] = n
            status.write(f"split   → {n} chunks")
            status.update(label="Building the search index...")
        elif name == "index":
            status.write(f"index   → {n} vectors ready")

    result = pipeline.build(load_text(), on_stage=on_stage)
    status.update(label=f"Ready — {state['chunks']} chunks indexed",
                  state="complete", expanded=False)
    st.session_state[key] = result
    return result


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


# --- sidebar (compact) ------------------------------------------------------
with st.sidebar:
    st.caption("PIPELINE — edit pipeline.py, save")
    st.code(
        f"split  = {pipeline.split.__module__.split('.')[-1]}\n"
        f"search = {pipeline.Search.__module__.split('.')[-1]}\n"
        f"chunk  = {pipeline.CHUNK_SIZE}/{pipeline.OVERLAP}\n"
        f"top_k  = {pipeline.TOP_K}\n"
        f"rerank = {pipeline.USE_RERANKER}",
        language="text",
    )
    uploaded = st.file_uploader("Ask document (PDF)", type=["pdf"], label_visibility="collapsed")
    st.caption("gemini: connected" if os.environ.get("GEMINI_API_KEY") else "gemini: not set (stub answers)")
    if FEEDBACK_FORM_URL:
        st.link_button("Feedback & scores", FEEDBACK_FORM_URL)

# --- build the Ask document once (shared by Ask and Index) ------------------
doc_path = DEFAULT_DOC
if uploaded is not None:
    doc_path = os.path.join(APP_DIR, f"_uploaded_{uploaded.name}")
    with open(doc_path, "wb") as fh:
        fh.write(uploaded.getvalue())
chunks, searcher = build_index(
    doc_path, lambda: load_pdf(doc_path), pipeline_sig(), "Indexing the document...")

st.title("Modern RAG in Practice")
st.markdown(pipeline_diagram_html(), unsafe_allow_html=True)
tab_ask, tab_race, tab_index = st.tabs(["Ask", "Race", "Index"])

with tab_ask:
    st.caption(f"{len(chunks)} chunks indexed from the document.")
    question = st.text_input("Your question", placeholder="e.g. How much does membership cost?")
    if st.button("Ask", type="primary") and question.strip():
        t0 = time.perf_counter()
        scored = retrieve_scored(searcher, question)
        t1 = time.perf_counter()
        rc = [rag.RetrievedChunk(text=c, score=s, index=i) for i, (c, s) in enumerate(scored)]
        used_llm = bool(os.environ.get("GEMINI_API_KEY"))

        st.subheader("Answer")
        st.write_stream(rag.stream_answer(question, rc))
        t2 = time.perf_counter()

        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{(t1 - t0) * 1000:.0f} ms")
        c2.metric("Generation", f"{(t2 - t1) * 1000:.0f} ms")
        c3.metric("Total", f"{(t2 - t0) * 1000:.0f} ms")
        st.caption(f"LLM used: {'yes' if used_llm else 'no (stub answer)'}")

        with st.expander("Final prompt sent to the LLM"):
            st.code(rag.build_prompt(question, rc), language="text")

        with st.expander("Sources", expanded=True):
            for rank, (chunk, s) in enumerate(scored, start=1):
                st.markdown(f"**Source {rank}** · similarity `{s:.3f}`")
                st.progress(max(0.0, min(1.0, float(s))))
                st.write(chunk)
                st.divider()

with tab_race:
    st.caption("Score the pipeline on the benchmark (120 questions, SQuAD-based). "
               "Edit pipeline.py, then run again. Keep TOP_K the same across the room.")
    if st.button("Run benchmark", type="primary"):
        _, bench_searcher = build_index(
            "benchmark", lambda: open(CORPUS, encoding="utf-8").read(),
            pipeline_sig(), "Indexing the benchmark corpus...")
        gold = load_gold(GOLD)

        def retrieve_fn(q):
            return [c for c, _ in retrieve_scored(bench_searcher, q)]

        generate_fn = None
        if os.environ.get("GEMINI_API_KEY"):
            def generate_fn(q, top):
                rc = [rag.RetrievedChunk(text=c, score=0.0, index=i) for i, c in enumerate(top)]
                return rag.generate_answer(q, rc)[0]

        bar = st.progress(0.0)
        line = st.empty()
        running = {"hits": 0}

        def on_progress(i, n, row):
            if row["found"]:
                running["hits"] += 1
            mark = f"rank {row['rank']}" if row["found"] else "missed"
            bar.progress(i / n)
            line.markdown(
                f"`{i:>3}/{n}`  hits `{running['hits']}`  ·  "
                f"{mark} — {row['question'][:70]}"
            )

        result = score(gold, retrieve_fn, generate_fn, on_progress=on_progress)
        bar.empty()
        line.empty()

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Recall@{pipeline.TOP_K}", f"{result['recall']:.3f}", f"{result['hits']}/{result['n']}", delta_color="off")
        c2.metric("MRR", f"{result['mrr']:.3f}")
        if result["answer_rate"] is not None:
            c3.metric(f"Answer@{pipeline.TOP_K}", f"{result['answer_rate']:.3f}")
        else:
            c3.caption("Answer@k needs a Gemini key")

        rows = result["rows"]
        buckets = {"rank 1": 0, "rank 2": 0, "rank 3+": 0, "missed": 0}
        for r in rows:
            if not r["found"]:
                buckets["missed"] += 1
            elif r["rank"] == 1:
                buckets["rank 1"] += 1
            elif r["rank"] == 2:
                buckets["rank 2"] += 1
            else:
                buckets["rank 3+"] += 1
        st.caption("Where the answer landed in your top-k:")
        st.bar_chart(pd.Series(buckets, name="questions"), color="#8a8a8a", horizontal=True)

        df = pd.DataFrame(rows)
        df["Rank"] = df["rank"].apply(lambda r: r if r else None)
        df = df.rename(columns={"question": "Question", "answer": "Answer", "found": "Found"})
        df = df.sort_values(by=["Found", "Rank"], ascending=[True, True], na_position="first")
        columns = ["Question", "Answer", "Found", "Rank"]
        colcfg = {
            "Question": st.column_config.TextColumn("Question", width="large"),
            "Found": st.column_config.CheckboxColumn("Found", help="answer appeared in your top-k chunks"),
            "Rank": st.column_config.NumberColumn("Rank", help="position of the first correct chunk (blank = missed)"),
        }
        if any(r["answer_hit"] is not None for r in rows):
            df["AI"] = df["answer_hit"]
            columns.append("AI")
            colcfg["AI"] = st.column_config.CheckboxColumn("AI", help="the LLM answer contained the gold phrase")
        st.caption("Every question (missed first):")
        st.dataframe(df[columns], hide_index=True, use_container_width=True, height=360, column_config=colcfg)

        label = f"{pipeline.split.__module__.split('.')[-1]}/{pipeline.Search.__module__.split('.')[-1]}"
        if pipeline.USE_RERANKER:
            label += "+rerank"
        line = f"{label:<28} Recall@{pipeline.TOP_K}={result['recall']:.3f}  MRR={result['mrr']:.3f}"
        if result["answer_rate"] is not None:
            line += f"  Answer@{pipeline.TOP_K}={result['answer_rate']:.3f}"
        st.caption("Copy your best line into benchmark/leaderboard.md:")
        st.code(line, language="text")
        if FEEDBACK_FORM_URL:
            st.link_button("Submit feedback & scores", FEEDBACK_FORM_URL)

with tab_index:
    st.caption(f"{len(chunks)} chunks · splitter={pipeline.split.__module__.split('.')[-1]} · "
               f"chunk={pipeline.CHUNK_SIZE}/{pipeline.OVERLAP}")
    lengths = [len(c) for c in chunks]
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks", len(chunks))
    m2.metric("Avg chars", sum(lengths) // len(lengths))
    m3.metric("Max chars", max(lengths))
    st.caption("Chunk length (characters):")
    st.bar_chart(pd.Series(lengths, name="chars"), color="#8a8a8a")
    idx = pd.DataFrame({
        "#": list(range(len(chunks))),
        "chars": lengths,
        "preview": [c[:160].replace("\n", " ") for c in chunks],
    })
    st.dataframe(idx, hide_index=True, use_container_width=True, height=360, column_config={
        "#": st.column_config.NumberColumn("#", width="small"),
        "chars": st.column_config.NumberColumn("chars", width="small"),
        "preview": st.column_config.TextColumn("preview", width="large"),
    })
