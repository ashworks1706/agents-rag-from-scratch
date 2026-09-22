"""Sentence splitting with NLTK: group whole sentences up to a size limit.

Splits text into sentences with NLTK, then packs sentences into chunks so a
chunk never ends mid-sentence.

pip install nltk
"""
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


if __name__ == "__main__":
    sample = ("Membership costs 15 dollars. Workshops run on Tuesday. "
              "Elections happen once a year. Dues can be waived for volunteers.")
    for i, c in enumerate(split(sample, chunk_size=60)):
        print(i, repr(c))
