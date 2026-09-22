"""Sentence splitting with spaCy: group sentences using a spaCy model.

Similar to the NLTK splitter but uses spaCy's sentence segmentation, which can
be more accurate on messy text.

pip install spacy   (the blank English pipeline used here needs no download)
"""
import spacy


def _nlp():
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def split(text, chunk_size=300):
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


if __name__ == "__main__":
    sample = ("Membership costs 15 dollars. Workshops run on Tuesday. "
              "Elections happen once a year. Dues can be waived for volunteers.")
    for i, c in enumerate(split(sample, chunk_size=60)):
        print(i, repr(c))
