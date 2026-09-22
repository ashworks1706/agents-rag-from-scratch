"""Token splitting: chunk by token count instead of characters.

Useful when you must respect a model's token limit exactly. Uses the tiktoken
encoding under the hood.

pip install langchain-text-splitters tiktoken
"""
from langchain_text_splitters import TokenTextSplitter


def split(text, chunk_size=128, overlap=16):
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


if __name__ == "__main__":
    sample = "The AI Society runs weekly workshops on machine learning and RAG. " * 5
    for i, c in enumerate(split(sample, chunk_size=20, overlap=4)):
        print(i, repr(c))
