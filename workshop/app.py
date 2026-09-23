"""Streamlit app for the Modern RAG workshop.

Pick your methods in the sidebar (no code editing) and the app re-indexes live.
Four tabs:
  Ask    - ask a question, see the answer, sources, and latency.
  Race   - score the pipeline on the benchmark (Recall@k / MRR).
  Index  - see the chunks the current document was split into.
  Learn  - every method, with a link to the deep-dive notebook.

pipeline.py still holds the defaults the sidebar starts from.
"""

import html
import importlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

from utils import load_pdf
from utils.benchmark import load_gold, score
import catalog
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
      section[data-testid="stSidebar"][aria-expanded="true"] { min-width: 260px; max-width: 300px; }
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

HAS_KEY = bool(os.environ.get("GEMINI_API_KEY"))

DEFAULTS = {
    "splitter": pipeline.split.__module__.split(".")[-1],
    "searcher": pipeline.Search.__module__.split(".")[-1],
    "chunk_size": pipeline.CHUNK_SIZE,
    "overlap": pipeline.OVERLAP,
    "top_k": pipeline.TOP_K,
    "use_reranker": pipeline.USE_RERANKER,
}


def build_sig(cfg):
    """What the index depends on (top_k / reranker are applied at query time)."""
    return (cfg["splitter"], cfg["searcher"], cfg["chunk_size"], cfg["overlap"])


def pipeline_diagram_html(cfg):
    """A left-to-right node diagram of the current pipeline."""

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
        node("split", cfg["splitter"], f"{cfg['chunk_size']}/{cfg['overlap']}"),
        node("search", cfg["searcher"], f"top {cfg['top_k']}"),
        node("rerank", "cross_encoder" if cfg["use_reranker"] else "off", dim=not cfg["use_reranker"]),
        node("answer", "gemini" if HAS_KEY else "stub", dim=not HAS_KEY),
    ]
    return (
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
        'margin:4px 0 16px;font-family:ui-monospace,SF Mono,Menlo,monospace">'
        + arrow.join(nodes) + "</div>"
    )


def tips_animation_html(tips, per=3.0):
    """CSS-only rotating tips. Runs client-side, so it keeps animating while the
    Python build blocks on embedding.
    """
    n = len(tips)
    total = n * per
    slot = 100.0 / n  # each tip's share of the full loop
    keyframes = (
        f"@keyframes tipcycle{{"
        f"0%{{opacity:0;transform:translateY(6px)}}"
        f"{slot*0.12:.2f}%{{opacity:1;transform:none}}"
        f"{slot*0.85:.2f}%{{opacity:1;transform:none}}"
        f"{slot:.2f}%{{opacity:0;transform:translateY(-6px)}}"
        f"100%{{opacity:0}}}}"
    )
    divs = "".join(
        f'<div class="tip" style="animation-delay:{i*per:.1f}s">{html.escape(t)}</div>'
        for i, t in enumerate(tips)
    )
    return (
        f"<style>{keyframes}"
        f".tips-wrap{{position:relative;height:40px;margin:2px 0 4px}}"
        f".tips-wrap .tip{{position:absolute;left:0;right:0;top:0;opacity:0;color:#b8b8b8;"
        f"font-size:13px;animation-name:tipcycle;animation-duration:{total:.1f}s;"
        f"animation-iteration-count:infinite;animation-timing-function:ease-in-out}}</style>"
        f'<div class="tips-wrap">{divs}</div>'
    )


def build_index(cache_key, load_text, sig, label, build_fn):
    """Build (chunks, searcher) once, narrating each real stage in a live
    status box. Cached in session_state, so a rebuild only runs when the
    pipeline signature changes. Returns None if the build fails.
    """
    key = f"index::{cache_key}::{sig}"
    if key in st.session_state:
        return st.session_state[key]

    box = st.empty()
    with box:
        status = st.status(label, expanded=True)
    status.markdown(tips_animation_html(catalog.TIPS), unsafe_allow_html=True)
    state = {"chunks": 0}

    def on_stage(name, n=None):
        if name == "split":
            state["chunks"] = n
            status.write(f"split   → {n} chunks")
            status.update(label="Building the search index...")
        elif name == "index":
            status.write(f"index   → {n} vectors ready")

    try:
        result = build_fn(load_text(), on_stage)
    except Exception as exc:
        status.update(label=f"Build failed: {exc.__class__.__name__}", state="error")
        status.write(str(exc))
        return None
    status.update(label=f"Ready — {state['chunks']} chunks indexed",
                  state="complete", expanded=False)
    st.session_state[key] = result
    return result


@st.cache_resource(show_spinner="Loading reranker...")
def get_reranker():
    from reranking.cross_encoder import Reranker
    return Reranker()


def retrieve_scored(searcher, query, cfg):
    n = cfg["top_k"] * 4 if cfg["use_reranker"] else cfg["top_k"]
    results = searcher.search(query, k=n)
    if isinstance(results, tuple):
        results = results[1]
    if cfg["use_reranker"]:
        candidates = [c for c, _ in results]
        results = get_reranker().rerank(query, candidates, k=cfg["top_k"])
    return results[:cfg["top_k"]]


# --- sidebar: edit the pipeline in the UI ------------------------------------
with st.sidebar:
    st.caption("PIPELINE — change it here, the app re-indexes live")
    splitters = list(catalog.SPLITTING)
    searchers = list(catalog.SEARCHING)
    cfg = {
        "splitter": st.selectbox(
            "split", splitters,
            index=splitters.index(DEFAULTS["splitter"]) if DEFAULTS["splitter"] in splitters else 0,
            key="cfg_splitter", help="how the document is cut into chunks"),
        "searcher": st.selectbox(
            "search", searchers,
            index=searchers.index(DEFAULTS["searcher"]) if DEFAULTS["searcher"] in searchers else 0,
            key="cfg_searcher", help="how chunks are retrieved for a query"),
    }
    c1, c2 = st.columns(2)
    cfg["chunk_size"] = c1.number_input("chunk", 100, 2000, DEFAULTS["chunk_size"], step=50,
                                        key="cfg_chunk", help=catalog.GLOSSARY["chunk"])
    cfg["overlap"] = c2.number_input("overlap", 0, 500, DEFAULTS["overlap"], step=10,
                                     key="cfg_overlap", help=catalog.GLOSSARY["overlap"])
    cfg["top_k"] = st.number_input("top_k", 1, 20, DEFAULTS["top_k"], step=1, key="cfg_topk",
                                   help=catalog.GLOSSARY["top_k"])
    cfg["use_reranker"] = st.toggle("reranker (cross-encoder)", value=DEFAULTS["use_reranker"],
                                    key="cfg_rerank", help=catalog.GLOSSARY["reranker"])

    st.divider()
    uploaded = st.file_uploader("Ask document (PDF)", type=["pdf"], label_visibility="collapsed")
    st.caption("gemini: connected" if HAS_KEY else "gemini: not set (stub answers)")
    if FEEDBACK_FORM_URL:
        st.link_button("Feedback & scores", FEEDBACK_FORM_URL)

# --- build the Ask document once (shared by Ask and Index) ------------------
doc_path = DEFAULT_DOC
if uploaded is not None:
    doc_path = os.path.join(APP_DIR, f"_uploaded_{uploaded.name}")
    with open(doc_path, "wb") as fh:
        fh.write(uploaded.getvalue())


def make_build_fn(cfg):
    return lambda text, on_stage=None: catalog.build_from_config(text, cfg, on_stage)


st.title("Modern RAG in Practice")
st.markdown(pipeline_diagram_html(cfg), unsafe_allow_html=True)
tab_ask, tab_race, tab_index, tab_learn = st.tabs(["Ask", "Race", "Index", "Learn"])

built = build_index(doc_path, lambda: load_pdf(doc_path), build_sig(cfg),
                    "Indexing the document...", make_build_fn(cfg))
if built is None:
    st.error(f"Could not build '{cfg['searcher']}' + '{cfg['splitter']}'. "
             "Some methods need extra packages — see requirements-modular.txt.")
    st.stop()
chunks, searcher = built

with tab_ask:
    st.caption(f"{len(chunks)} chunks indexed from the document.")
    question = st.text_input("Your question", placeholder="e.g. How much does membership cost?")
    if st.button("Ask", type="primary") and question.strip():
        t0 = time.perf_counter()
        with st.spinner("Retrieving relevant chunks..."):
            scored = retrieve_scored(searcher, question, cfg)
        t1 = time.perf_counter()
        rc = [rag.RetrievedChunk(text=c, score=s, index=i) for i, (c, s) in enumerate(scored)]

        st.subheader("Answer")
        with st.spinner("Generating answer..." if HAS_KEY else "Preparing..."):
            st.write_stream(rag.stream_answer(question, rc))
        t2 = time.perf_counter()

        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{(t1 - t0) * 1000:.0f} ms")
        c2.metric("Generation", f"{(t2 - t1) * 1000:.0f} ms")
        c3.metric("Total", f"{(t2 - t0) * 1000:.0f} ms")
        st.caption(f"LLM used: {'yes' if HAS_KEY else 'no (stub answer)'}")

        with st.expander("Final prompt sent to the LLM"):
            st.code(rag.build_prompt(question, rc), language="text")

        with st.expander("Sources", expanded=True):
            for rank, (chunk, s) in enumerate(scored, start=1):
                st.markdown(f"**Source {rank}** · similarity `{s:.3f}`")
                st.progress(max(0.0, min(1.0, float(s))))
                st.write(chunk)
                st.divider()

with tab_race:
    st.caption("Score the pipeline on the benchmark (SQuAD-based). "
               "Change methods in the sidebar, then run again. Keep top_k the same across the room.")
    mode = st.radio(
        "Benchmark size", ["Quick test (20)", "Full (120)"], horizontal=True,
        help="Quick runs an evenly-spread 20-question sample — good while tuning. "
             "Run Full for the score you report.")
    run_col, coach_col = st.columns([1, 1.3])
    run = run_col.button("Run benchmark", type="primary")
    with coach_col.popover("Copy coach prompt"):
        st.caption("Paste this into ChatGPT or Claude. It coaches you through the "
                   "challenge on principles — it won't just hand you the best config. "
                   "Use the copy icon in the corner.")
        st.code(catalog.coach_prompt(cfg), language="text")
    if run:
        bench = build_index("benchmark", lambda: open(CORPUS, encoding="utf-8").read(),
                            build_sig(cfg), "Indexing the benchmark corpus...", make_build_fn(cfg))
        if bench is None:
            st.error("Could not build this pipeline on the benchmark corpus.")
            st.stop()
        _, bench_searcher = bench
        gold = load_gold(GOLD)
        if mode.startswith("Quick"):
            step = max(1, len(gold) // 20)
            gold = gold[::step][:20]

        def retrieve_fn(q):
            return [c for c, _ in retrieve_scored(bench_searcher, q, cfg)]

        generate_fn = None
        if HAS_KEY:
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
        c1.metric(f"Recall@{cfg['top_k']}", f"{result['recall']:.3f}", f"{result['hits']}/{result['n']}",
                  delta_color="off", help=catalog.GLOSSARY["recall"])
        c2.metric("MRR", f"{result['mrr']:.3f}", help=catalog.GLOSSARY["mrr"])
        if result["answer_rate"] is not None:
            c3.metric(f"Answer@{cfg['top_k']}", f"{result['answer_rate']:.3f}", help=catalog.GLOSSARY["answer_rate"])
        else:
            c3.caption("Answer@k needs a Gemini key")
        if mode.startswith("Quick"):
            st.caption(f"Quick sample of {result['n']} — run Full (120) before reporting a score.")

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

        label = f"{cfg['splitter']}/{cfg['searcher']}"
        if cfg["use_reranker"]:
            label += "+rerank"
        best = f"{label:<28} Recall@{cfg['top_k']}={result['recall']:.3f}  MRR={result['mrr']:.3f}"
        if result["answer_rate"] is not None:
            best += f"  Answer@{cfg['top_k']}={result['answer_rate']:.3f}"
        st.caption("Copy your best line into benchmark/leaderboard.md:")
        st.code(best, language="text")
        if FEEDBACK_FORM_URL:
            st.link_button("Submit feedback & scores", FEEDBACK_FORM_URL)

with tab_index:
    st.caption(f"{len(chunks)} chunks · splitter={cfg['splitter']} · "
               f"chunk={cfg['chunk_size']}/{cfg['overlap']}")
    lengths = [len(c) for c in chunks]
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks", len(chunks), help=catalog.GLOSSARY["chunk"])
    m2.metric("Avg chars", sum(lengths) // len(lengths), help="Average chunk length in characters.")
    m3.metric("Max chars", max(lengths), help="Longest chunk in characters.")
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

with tab_learn:
    st.caption("Every method in the library, with a link to the section of the "
               "deep-dive notebook that explains it. Pick methods in the sidebar.")
    selected = {"split": cfg["splitter"], "search": cfg["searcher"]}
    for stage, selectable, folder, methods in catalog.STAGES:
        head = f"{stage} · `{folder}/`"
        if not selectable:
            head += " · reference (set by the search method)"
        st.subheader(head)
        for name, (blurb, cell_id) in methods.items():
            mark = " — **selected**" if selected.get(stage) == name else ""
            st.markdown(
                f"- **{name}**{mark} — {blurb} "
                f"[notebook ↗]({catalog.colab_link(cell_id)})"
            )
