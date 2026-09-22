"""Language-aware code splitting: split source code on syntax boundaries.

Uses separators specific to a programming language (functions, classes) so
chunks do not cut through the middle of a construct.

pip install langchain-text-splitters
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def split(code, language=Language.PYTHON, chunk_size=300, overlap=0):
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language, chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(code)


if __name__ == "__main__":
    sample = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"
    for i, c in enumerate(split(sample, chunk_size=40)):
        print(i, repr(c))
