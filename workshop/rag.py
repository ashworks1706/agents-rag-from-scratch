"""Minimal RAG pipeline: load -> chunk -> embed -> cosine top-k -> LLM answer.

No framework. Set GEMINI_API_KEY to enable generation; without it, retrieval,
scores and latency still work and the answer is a labeled stub.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 3


@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


@dataclass
class RagResult:
    answer: str
    chunks: list[RetrievedChunk]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    used_llm: bool
    query: str = ""
    warnings: list[str] = field(default_factory=list)


def load_pdf(path: str) -> str:
    import pymupdf
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = " ".join(text.split())
    step = chunk_size - overlap
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def embed(texts: list[str]) -> np.ndarray:
    # Normalise so a dot product equals cosine similarity.
    vecs = get_embedder().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vecs / norms


def cosine_top_k(query_vec: np.ndarray, chunk_vecs: np.ndarray, chunks: list[str], k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
    scores = chunk_vecs @ query_vec
    top_idx = np.argsort(-scores)[: min(k, len(chunks))]
    return [RetrievedChunk(chunks[i], float(scores[i]), int(i)) for i in top_idx]


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
        f"Most relevant passage:\n\n\"{top[:300]}...\""
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, bool, Optional[str]]:
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
    except Exception as exc:
        return _stub_answer(chunks), False, f"Generation failed ({exc.__class__.__name__}): {exc}"


class RagPipeline:
    def __init__(self, chunks: list[str], chunk_vecs: np.ndarray):
        self.chunks = chunks
        self.chunk_vecs = chunk_vecs

    @classmethod
    def from_pdf(cls, path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> "RagPipeline":
        return cls.from_text(load_pdf(path), chunk_size, overlap)

    @classmethod
    def from_text(cls, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> "RagPipeline":
        chunks = chunk_text(text, chunk_size, overlap)
        return cls(chunks, embed(chunks))

    def answer(self, question: str, k: int = DEFAULT_TOP_K) -> RagResult:
        warnings: list[str] = []
        t0 = time.perf_counter()
        retrieved = cosine_top_k(embed([question])[0], self.chunk_vecs, self.chunks, k)
        t1 = time.perf_counter()
        answer, used_llm, warning = generate_answer(question, retrieved)
        t2 = time.perf_counter()
        if warning:
            warnings.append(warning)
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
    pipe = RagPipeline.from_pdf("sample_document.pdf")
    res = pipe.answer("How much does membership cost?")
    print("USED LLM:", res.used_llm)
    print("ANSWER:", res.answer[:400])
    print(f"retrieval {res.retrieval_ms:.1f}ms  generation {res.generation_ms:.1f}ms  total {res.total_ms:.1f}ms")
