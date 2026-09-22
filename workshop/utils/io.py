"""Small helpers so main.py stays as just the pipeline steps."""

import sys


def load_pdf(path):
    """Return all text from a PDF."""
    import pymupdf
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def get_query(default):
    """First command-line argument, or a default question."""
    return sys.argv[1] if len(sys.argv) > 1 else default


def print_results(query, chunks, results):
    """Print the final ranked chunks."""
    print(f"Query: {query}\n")
    print(f"Indexed {len(chunks)} chunks; showing top {len(results)}.\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.3f}  {chunk[:100]}...")
