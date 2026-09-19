"""
app.py — Streamlit interface for the Modern RAG in Practice workshop.

Attendees do NOT write this file live; they read it and deploy it. It imports
the exact pipeline they built in the notebook (rag.py) and wraps it in a UI
with the observability panel: retrieved chunks, per-chunk similarity scores,
and retrieval / generation / total latency.

Run locally:   streamlit run app.py
Deploy:        push to GitHub -> Streamlit Community Cloud -> point at workshop/app.py
"""

import os
import streamlit as st

import rag

st.set_page_config(page_title="Modern RAG in Practice", page_icon="🔎", layout="centered")

# --- Bridge Streamlit secrets -> env var so rag.py finds the key -------------
if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

DEFAULT_DOC = "sample_document.pdf"


@st.cache_resource(show_spinner="Indexing the document (embeddings)...")
def build_pipeline_from_pdf(path: str, chunk_size: int, overlap: int):
    """Cached so the model + embeddings are computed once per container."""
    return rag.RagPipeline.from_pdf(path, chunk_size=chunk_size, overlap=overlap)


@st.cache_resource(show_spinner="Indexing the uploaded document...")
def build_pipeline_from_bytes(data: bytes, name: str, chunk_size: int, overlap: int):
    tmp_path = os.path.join(st.session_state.get("_tmp_dir", "."), f"_uploaded_{name}")
    with open(tmp_path, "wb") as f:
        f.write(data)
    return rag.RagPipeline.from_pdf(tmp_path, chunk_size=chunk_size, overlap=overlap)


st.title("🔎 Modern RAG in Practice")
st.caption("Ask a question about the document. Every answer shows its sources, scores, and latency.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, rag.DEFAULT_TOP_K)
    chunk_size = st.slider("Chunk size (chars)", 200, 1200, rag.DEFAULT_CHUNK_SIZE, step=50)
    overlap = st.slider("Chunk overlap (chars)", 0, 300, rag.DEFAULT_CHUNK_OVERLAP, step=25)

    st.divider()
    st.subheader("Document")
    st.write("Default: **AI Society Handbook** (sample).")
    uploaded = st.file_uploader("Or upload your own PDF (extension)", type=["pdf"])

    st.divider()
    if os.environ.get("GEMINI_API_KEY"):
        st.success("Gemini key detected — answers are generated.")
    else:
        st.warning("No Gemini key — retrieval works; answer is a stub. "
                   "Add GEMINI_API_KEY in Settings → Secrets.")

# Build (or rebuild) the pipeline for the chosen document + settings.
if uploaded is not None:
    pipe = build_pipeline_from_bytes(uploaded.getvalue(), uploaded.name, chunk_size, overlap)
    doc_label = uploaded.name
else:
    if not os.path.exists(DEFAULT_DOC):
        st.error(f"Sample document '{DEFAULT_DOC}' not found next to app.py.")
        st.stop()
    pipe = build_pipeline_from_pdf(DEFAULT_DOC, chunk_size, overlap)
    doc_label = "AI Society Handbook (sample)"

st.info(f"Indexed **{len(pipe.chunks)}** chunks from *{doc_label}*.")

question = st.text_input("Your question", placeholder="e.g. How much does membership cost?")
ask = st.button("Ask", type="primary")

if ask and question.strip():
    with st.spinner("Retrieving and generating..."):
        result = pipe.answer(question, k=top_k)

    st.subheader("Answer")
    st.write(result.answer)

    for w in result.warnings:
        st.caption(f"⚠️ {w}")

    with st.expander("How was this answer generated?", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval", f"{result.retrieval_ms:.0f} ms")
        c2.metric("Generation", f"{result.generation_ms:.0f} ms")
        c3.metric("Total", f"{result.total_ms:.0f} ms")
        st.caption(f"LLM used: {'yes' if result.used_llm else 'no (stub answer)'}")

        st.markdown("**Retrieved chunks (sources)**")
        for rank, chunk in enumerate(result.chunks, start=1):
            st.markdown(f"**Source {rank}** · chunk #{chunk.index} · similarity `{chunk.score:.3f}`")
            st.progress(max(0.0, min(1.0, chunk.score)))
            st.write(chunk.text)
            st.divider()

elif ask:
    st.warning("Type a question first.")
