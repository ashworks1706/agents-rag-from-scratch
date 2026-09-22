"""Answer generation for the app: build a grounded prompt from retrieved chunks
and call Gemini. Retrieval lives in the modular stage folders; this file only
turns retrieved chunks into an answer (streaming or one-shot), with a no-key
stub fallback.
"""

import os
from dataclasses import dataclass

GEMINI_MODEL_NAME = "gemini-3.6-flash"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a document.
Use ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RetrievedChunk:
    text: str
    score: float
    index: int


def build_prompt(question, chunks):
    """The exact augmented prompt sent to the LLM."""
    context = "\n\n---\n\n".join(f"[Source {i+1}] {c.text}" for i, c in enumerate(chunks))
    return PROMPT_TEMPLATE.format(context=context, question=question)


def _stub_answer(chunks):
    top = chunks[0].text if chunks else "(no chunks retrieved)"
    return (
        "[No GEMINI_API_KEY set. Showing retrieval only, no generated answer.]\n\n"
        f"Most relevant passage:\n\n\"{top[:300]}...\""
    )


def generate_answer(question, chunks):
    """Return (answer, used_llm, warning). One-shot; used by the benchmark."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _stub_answer(chunks), False, "No GEMINI_API_KEY set; generation skipped."
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=build_prompt(question, chunks))
        return (resp.text or "").strip(), True, None
    except Exception as exc:
        return _stub_answer(chunks), False, f"Generation failed ({exc.__class__.__name__}): {exc}"


def stream_answer(question, chunks):
    """Yield answer text as it streams from the LLM (for st.write_stream).
    Without a key, yields the stub once.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        yield _stub_answer(chunks)
        return
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for event in client.models.generate_content_stream(model=GEMINI_MODEL_NAME, contents=build_prompt(question, chunks)):
            if getattr(event, "text", None):
                yield event.text
    except Exception as exc:
        yield f"[Generation failed ({exc.__class__.__name__}): {exc}]"
