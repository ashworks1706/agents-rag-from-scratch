"""LaTeX splitting: split a LaTeX document on its structural commands.

Splits on sections, subsections, and environments so equations and their
surrounding text stay together.

pip install langchain-text-splitters
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def split(latex_text, chunk_size=300, overlap=0):
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.LATEX, chunk_size=chunk_size, chunk_overlap=overlap
    )
    return splitter.split_text(latex_text)


if __name__ == "__main__":
    sample = r"\section{Intro}Text here.\subsection{Method}More text.\begin{equation}E=mc^2\end{equation}"
    for i, c in enumerate(split(sample, chunk_size=40)):
        print(i, repr(c))
