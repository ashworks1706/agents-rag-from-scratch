"""RAG pipeline built on LangChain with a FAISS vector store.

Flow: load PDF -> split -> embed -> FAISS similarity search -> LLM answer.
Set GEMINI_API_KEY to enable generation; without it, retrieval, scores and
latency still work and the answer is a labeled stub.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL_NAME = "gemini-3.6-flash"
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


_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        # Normalised vectors so inner product equals cosine similarity.
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _build_store(docs):
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    return FAISS.from_documents(
        docs,
        get_embeddings(),
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a document.
Use ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def _stub_answer(chunks: list[RetrievedChunk]) -> str:
    top = chunks[0].text if chunks else "(no chunks retrieved)"
    return (
        "[No GEMINI_API_KEY set. Showing retrieval only, no generated answer.]\n\n"
        f"Most relevant passage:\n\n\"{top[:300]}...\""
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> tuple[str, bool, Optional[str]]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _stub_answer(chunks), False, "No GEMINI_API_KEY set; generation skipped."
    context = "\n\n---\n\n".join(f"[Source {i+1}] {c.text}" for i, c in enumerate(chunks))
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=api_key)
        return llm.invoke(prompt).content.strip(), True, None
    except Exception as exc:
        return _stub_answer(chunks), False, f"Generation failed ({exc.__class__.__name__}): {exc}"


class RagPipeline:
    def __init__(self, store):
        self.store = store

    @classmethod
    def from_pdf(cls, path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> "RagPipeline":
        from langchain_community.document_loaders import PyMuPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        pages = PyMuPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        docs = splitter.split_documents(pages)
        return cls(_build_store(docs))

    @classmethod
    def from_text(cls, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> "RagPipeline":
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        docs = splitter.split_documents([Document(page_content=text)])
        return cls(_build_store(docs))

    @property
    def chunks(self) -> list[str]:
        store = self.store.docstore._dict
        return [d.page_content for d in store.values()]

    def answer(self, question: str, k: int = DEFAULT_TOP_K) -> RagResult:
        warnings: list[str] = []
        t0 = time.perf_counter()
        hits = self.store.similarity_search_with_score(question, k=k)
        retrieved = [
            RetrievedChunk(text=doc.page_content, score=float(score), index=i)
            for i, (doc, score) in enumerate(hits)
        ]
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
