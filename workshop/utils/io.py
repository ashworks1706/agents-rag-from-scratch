"""Small shared helpers."""


def load_pdf(path):
    """Return all text from a PDF."""
    import pymupdf
    doc = pymupdf.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text
