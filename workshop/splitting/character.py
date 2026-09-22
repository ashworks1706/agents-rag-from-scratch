"""Character splitting: cut text on a separator into fixed-size chunks.

The simplest splitter. Splits on a separator (default: blank lines), then packs
pieces up to chunk_size with some overlap. Good baseline; ignores meaning.

pip install langchain-text-splitters
"""
from langchain_text_splitters import CharacterTextSplitter


def split(text, chunk_size=500, overlap=100, separator="\n\n"):
    splitter = CharacterTextSplitter(
        separator=separator, chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    sample = ("Membership costs 15 dollars per semester.\n\n"
              "Workshops run every Tuesday from 6 to 8 PM.\n\n"
              "Officer elections are held once per year.")
    for i, c in enumerate(split(sample, chunk_size=60, overlap=10)):
        print(i, repr(c))
