"""Language-aware code splitting: split source code on syntax boundaries (functions, classes) for a given language."""
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def split(code, language=Language.PYTHON, chunk_size=300, overlap=0):
    return RecursiveCharacterTextSplitter.from_language(language=language, chunk_size=chunk_size, chunk_overlap=overlap).split_text(code)
