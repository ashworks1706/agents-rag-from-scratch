"""HTML header splitting: split an HTML document by its heading tags.

Like the Markdown splitter but for HTML (h1/h2/h3). Returns section texts.

pip install langchain-text-splitters lxml
"""
from langchain_text_splitters import HTMLHeaderTextSplitter

HEADERS = [("h1", "h1"), ("h2", "h2"), ("h3", "h3")]


def split(html_text, headers=HEADERS):
    splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
    return [d.page_content for d in splitter.split_text(html_text)]


if __name__ == "__main__":
    sample = "<h1>Handbook</h1><h2>Dues</h2><p>Membership costs 15 dollars.</p><h2>Workshops</h2><p>Every Tuesday.</p>"
    for i, c in enumerate(split(sample)):
        print(i, repr(c))
