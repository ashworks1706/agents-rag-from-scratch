"""Sentence splitting with spaCy: segment sentences with a blank English pipeline, then pack them up to chunk_size."""
import spacy


def _nlp():
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def split(text, chunk_size=300, overlap=0):
    doc = _nlp()(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
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
