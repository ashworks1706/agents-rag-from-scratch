"""Recursive splitting: try separators in order (paragraphs, lines, spaces, characters) until chunks fit chunk_size. The common default."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split(text, chunk_size=500, overlap=100):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)
