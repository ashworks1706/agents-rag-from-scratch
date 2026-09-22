"""Character splitting: cut text on a separator, then pack pieces up to chunk_size with overlap. The simplest splitter; ignores meaning."""
from langchain_text_splitters import CharacterTextSplitter


def split(text, chunk_size=500, overlap=100, separator="\n\n"):
    return CharacterTextSplitter(separator=separator, chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)
