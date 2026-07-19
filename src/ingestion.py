"""
Ingestion pipeline: load documents (files or URLs) -> chunk -> embed -> store in Chroma.

Chunking strategy
------------------
Technical documentation mixes prose paragraphs with code blocks and markdown headers.
Splitting purely by character count risks cutting a code block in half, which destroys
its meaning and makes it useless as retrieved context. To avoid that we use LangChain's
`MarkdownHeaderTextSplitter` first (so each chunk keeps its section header as metadata,
which improves both grading and citation quality), and then run a
`RecursiveCharacterTextSplitter` configured with markdown-aware separators
(headers > code fences > paragraphs > sentences > words) as a second pass, so no
individual chunk exceeds CHUNK_SIZE tokens. CHUNK_OVERLAP (~15% of chunk size) preserves
continuity for statements that span a chunk boundary, e.g. a sentence introducing a code
block that starts in the next chunk.
"""
import os
from pathlib import Path

from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from typing import List, Optional
from src import config

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def load_local_file(path: str) -> Document:
    text = Path(path).read_text(encoding="utf-8")
    return Document(page_content=text, metadata={"source": Path(path).name})


def load_from_url(url: str) -> Document:
    import requests

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")

    if "html" in content_type:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator="\n")
    else:
        text = resp.text

    name = Path(urlparse(url).path).name or url
    return Document(page_content=text, metadata={"source": name, "url": url})


def load_corpus_dir(directory: str) -> List[Document]:
    docs = []
    for path in sorted(Path(directory).glob("*")):
        if path.suffix.lower() in (".md", ".txt", ".html"):
            docs.append(load_local_file(str(path)))
    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Two-pass split: markdown headers first, then size-bounded recursive split."""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n```", "\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: List[Document] = []
    for doc in documents:
        try:
            header_chunks = md_splitter.split_text(doc.page_content)
        except Exception:
            header_chunks = [doc]

        for hc in header_chunks:
            # Carry over original source metadata, plus any header metadata
            merged_meta = {**doc.metadata, **getattr(hc, "metadata", {})}
            hc.metadata = merged_meta

        sized_chunks = char_splitter.split_documents(header_chunks)
        all_chunks.extend(sized_chunks)

    # Attach a stable chunk id for citation purposes
    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}::{i}"

    return all_chunks


def ingest_paths(
    paths: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
) -> int:
    """Ingest local files and/or URLs into the vector store. Returns number of chunks added."""
    from src.vectorstore import add_documents

    documents: List[Document] = []
    for p in paths or []:
        documents.append(load_local_file(p))
    for u in urls or []:
        documents.append(load_from_url(u))

    if not documents:
        return 0

    chunks = chunk_documents(documents)
    add_documents(chunks)
    return len(chunks)


def ingest_corpus_dir(directory: Optional[str] = None) -> int:
    """Ingest every supported file in a directory (used for the initial/startup ingest)."""
    from src.vectorstore import add_documents

    directory = directory or config.CORPUS_DIR
    documents = load_corpus_dir(directory)
    if not documents:
        return 0

    chunks = chunk_documents(documents)
    add_documents(chunks)
    return len(chunks)
