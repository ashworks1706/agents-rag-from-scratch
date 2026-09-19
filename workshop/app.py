"""Streamlit interface for the RAG pipeline in rag.py."""

import os
import streamlit as st

import rag

st.set_page_config(page_title="Modern RAG in Practice", page_icon="🔎", layout="centered")

if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

DEFAULT_DOC = "sample_document.pdf"


@st.cache_resource(show_spinner="Indexing the document...")
def build_from_pdf(path, chunk_size, overlap):
    return rag.RagPipeline.from_pdf(path, chunk_size=chunk_size, overlap=overlap)


@st.cache_resource(show_spinner="Indexing the uploaded document...")
def build_from_bytes(data, name, chunk_size, overlap):
    tmp = f"_uploaded_{name}"
    with open(tmp, "wb") as f:
        f.write(data)
    return rag.RagPipeline.from_pdf(tmp, chunk_size=chunk_size, overlap=overlap)


st.title("🔎 Modern RAG in Practice")
st.caption("Ask a question about the document. Every answer shows its sources, scores, and latency.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 8, rag.DEFAULT_TOP_K)
    chunk_size = st.slider("Chunk size (chars)", 200, 1200, rag.DEFAULT_CHUNK_SIZE, step=50)
    overlap = st.slider("Chunk overlap (chars)", 0, 300, rag.DEFAULT_CHUNK_OVERLAP, step=25)

    st.divider()
    uploaded = st.file_uploader("Upload a PDF (optional)", type=["pdf"])

    st.divider()
    if os.environ.get("GEMINI_API_KEY"):
        st.success("Gemini key detected — answers are generated.")
    else:
        st.warning("No Gemini key — retrieval works; answer is a stub. Add GEMINI_API_KEY in Settings → Secrets.")

if uploaded is not None:
    pipe = build_from_bytes(uploaded.getvalue(), uploaded.name, chunk_size, overlap)
    doc_label = uploaded.name
else:
    if not os.path.exists(DEFAULT_DOC):
        st.error(f"'{DEFAULT_DOC}' not found next to app.py.")
        st.stop()
    pipe = build_from_pdf(DEFAULT_DOC, chunk_size, overlap)
    doc_label = "sample document"

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
