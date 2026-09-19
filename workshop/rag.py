"""
rag.py — a minimal, transparent RAG pipeline for the "Modern RAG in Practice" workshop.

One continuous path, no framework:
    load PDF -> chunk -> embed -> cosine similarity -> top-k -> LLM answer -> sources + metrics

Everything here is plain Python + NumPy + Sentence Transformers + the Gemini API.
No LangChain, no vector database. That is the point: you can read every line.

Set GEMINI_API_KEY in the environment (or Streamlit secrets) to enable answer
generation. Without a key, retrieval, scores, and latency still work and the
answer is a clearly-labeled stub — so the retrieval half of the workshop never
depends on an API.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, CPU-friendly
GEMINI_MODEL_NAME = "gemini-2.0-flash"  # free-tier chat model; change here if renamed
DEFAULT_CHUNK_SIZE = 500                 # characters per chunk
DEFAULT_CHUNK_OVERLAP = 100              # characters shared between neighbours
DEFAULT_TOP_K = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


@dataclass
class RagResult:
    """Everything the UI needs, including the observability numbers."""
    answer: str
    chunks: list[RetrievedChunk]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    used_llm: bool
    query: str = ""
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------

def load_pdf(path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    import pymupdf  # PyMuPDF (older code imports this as `fitz`)
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


# ---------------------------------------------------------------------------
# 2. Chunk
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Fixed-size character chunks with overlap.

    Overlap keeps a sentence that straddles a boundary retrievable from either
    side. This is the simplest strategy that works; production systems get
    fancier, but the mechanism is identical.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = " ".join(text.split())  # normalise whitespace
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# 3. Embed
# ---------------------------------------------------------------------------

_embedder = None


def get_embedder():
    """Load the sentence-transformer once and reuse it."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings, shape (n, dim).

    Normalising means a dot product IS the cosine similarity, so retrieval is
    one matrix multiply.
    """
    model = get_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vecs / norms


# ---------------------------------------------------------------------------
# 4 & 5. Cosine similarity + top-k retrieval
# ---------------------------------------------------------------------------

def cosine_top_k(
    query_vec: np.ndarray,
    chunk_vecs: np.ndarray,
    chunks: list[str],
    k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Rank chunks by cosine similarity to the query and return the top k.

    Both inputs are already normalised, so the dot product is the cosine.
    """
    scores = chunk_vecs @ query_vec  # (n,)
    k = min(k, len(chunks))
    top_idx = np.argsort(-scores)[:k]
    return [
        RetrievedChunk(text=chunks[i], score=float(scores[i]), index=int(i))
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# 6. Generate
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a document.
Use ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def _stub_answer(chunks: list[RetrievedChunk]) -> str:
    top = chunks[0].text if chunks else "(no chunks retrieved)"
    return (
        "[No GEMINI_API_KEY set — showing retrieval only, no generated answer.]\n\n"
        "The most relevant passage the retriever found was:\n\n"
        f"\"{top[:300]}...\""
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, bool, Optional[str]]:
    """Call Gemini with the retrieved context. Returns (answer, used_llm, warning)."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _stub_answer(chunks), False, "No GEMINI_API_KEY set; generation skipped."

    context = "\n\n---\n\n".join(f"[Source {i+1}] {c.text}" for i, c in enumerate(chunks))
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
        return (resp.text or "").strip(), True, None
    except Exception as exc:  # network, quota, bad key, renamed model
        return _stub_answer(chunks), False, f"Generation failed ({exc.__class__.__name__}): {exc}"


# ---------------------------------------------------------------------------
# Orchestrator — the whole pipeline, timed
# ---------------------------------------------------------------------------

class RagPipeline:
    """Holds the indexed document so queries are cheap to run repeatedly."""

    def __init__(self, chunks: list[str], chunk_vecs: np.ndarray):
        self.chunks = chunks
        self.chunk_vecs = chunk_vecs

    @classmethod
    def from_pdf(
        cls,
        path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> "RagPipeline":
        text = load_pdf(path)
        chunks = chunk_text(text, chunk_size, overlap)
        chunk_vecs = embed(chunks)
        return cls(chunks, chunk_vecs)

    @classmethod
    def from_text(
        cls,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> "RagPipeline":
        chunks = chunk_text(text, chunk_size, overlap)
        chunk_vecs = embed(chunks)
        return cls(chunks, chunk_vecs)

    def answer(self, question: str, k: int = DEFAULT_TOP_K) -> RagResult:
        warnings: list[str] = []
        t0 = time.perf_counter()

        query_vec = embed([question])[0]
        retrieved = cosine_top_k(query_vec, self.chunk_vecs, self.chunks, k)
        t1 = time.perf_counter()

        answer, used_llm, warning = generate_answer(question, retrieved)
        if warning:
            warnings.append(warning)
        t2 = time.perf_counter()

        return RagResult(
            answer=answer,
            chunks=retrieved,
            retrieval_ms=(t1 - t0) * 1000,
            generation_ms=(t2 - t1) * 1000,
            total_ms=(t2 - t0) * 1000,
            used_llm=used_llm,
            query=question,
            warnings=warnings,
        )


if __name__ == "__main__":
    # Smoke test against the sample document.
    pipe = RagPipeline.from_pdf("sample_document.pdf")
    res = pipe.answer("How much does membership cost?")
    print("USED LLM:", res.used_llm)
    print("ANSWER:", res.answer[:400])
    print(f"retrieval {res.retrieval_ms:.1f}ms  generation {res.generation_ms:.1f}ms  total {res.total_ms:.1f}ms")
    for c in res.chunks:
        print(f"  [{c.index}] score={c.score:.3f} {c.text[:70]}...")
