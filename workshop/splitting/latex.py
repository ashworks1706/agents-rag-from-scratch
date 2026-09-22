"""LaTeX splitting: split on LaTeX structure (sections, environments) so equations stay with their surrounding text."""
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def split(latex_text, chunk_size=300, overlap=0):
    return RecursiveCharacterTextSplitter.from_language(language=Language.LATEX, chunk_size=chunk_size, chunk_overlap=overlap).split_text(latex_text)
