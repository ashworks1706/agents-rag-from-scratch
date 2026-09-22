"""Recursive character splitting: the common default.

Tries a list of separators in order (paragraphs, lines, spaces, characters)
until the chunks fit chunk_size. Keeps related text together better than a
single-separator split.

pip install langchain-text-splitters
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    sample = ("Membership costs 15 dollars per semester. Workshops run every "
              "Tuesday from 6 to 8 PM in room 210. Officer elections are held "
              "once per year during the last spring workshop.")
    for i, c in enumerate(split(sample, chunk_size=80, overlap=20)):
        print(i, repr(c))
