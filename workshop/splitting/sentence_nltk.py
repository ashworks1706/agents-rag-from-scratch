"""Sentence splitting with NLTK: split into sentences, then pack them up to chunk_size so a chunk never ends mid-sentence."""
import nltk


def _ensure_punkt():
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
                return
            except Exception:
                continue


def split(text, chunk_size=300):
    _ensure_punkt()
    sentences = nltk.sent_tokenize(text)
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > chunk_size:
            chunks.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current.strip())
    return chunks
