"""Markdown header splitting: split a Markdown document by its headings.

Keeps each section together and attaches the heading path as metadata. Returns
the section texts.

pip install langchain-text-splitters
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def split(markdown_text, headers=HEADERS):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    return [d.page_content for d in splitter.split_text(markdown_text)]


if __name__ == "__main__":
    sample = "# Handbook\n\n## Dues\nMembership costs 15 dollars.\n\n## Workshops\nEvery Tuesday."
    for i, c in enumerate(split(sample)):
        print(i, repr(c))
