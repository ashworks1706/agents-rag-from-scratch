"""Markdown header splitting: split a Markdown document by its headings so each section stays together."""
from langchain_text_splitters import MarkdownHeaderTextSplitter

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def split(markdown_text, headers=HEADERS):
    return [d.page_content for d in MarkdownHeaderTextSplitter(headers_to_split_on=headers).split_text(markdown_text)]
