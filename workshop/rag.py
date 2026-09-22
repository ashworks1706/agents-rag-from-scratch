"""Answer generation for the app: build a grounded prompt from retrieved chunks
and call Gemini. Retrieval itself lives in the modular stage folders; this file
only turns retrieved chunks into an answer, with a no-key stub fallback.
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


def _stub_answer(chunks):
    top = chunks[0].text if chunks else "(no chunks retrieved)"
    return (
        "[No GEMINI_API_KEY set. Showing retrieval only, no generated answer.]\n\n"
        f"Most relevant passage:\n\n\"{top[:300]}...\""
    )


def generate_answer(question, chunks):
    """Return (answer, used_llm, warning)."""
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
