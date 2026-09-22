"""HTML header splitting: split an HTML document by its heading tags (h1/h2/h3) so each section stays together."""
from langchain_text_splitters import HTMLHeaderTextSplitter

HEADERS = [("h1", "h1"), ("h2", "h2"), ("h3", "h3")]


def split(html_text, headers=HEADERS):
    return [d.page_content for d in HTMLHeaderTextSplitter(headers_to_split_on=headers).split_text(html_text)]
