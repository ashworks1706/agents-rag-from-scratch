"""Token splitting: chunk by token count (tiktoken) instead of characters, to respect a model's token limit exactly."""
from langchain_text_splitters import TokenTextSplitter


def split(text, chunk_size=128, overlap=16):
    return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)
