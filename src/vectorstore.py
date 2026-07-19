"""
FAISS-backed vector store.

Persistence is manual with FAISS -- LangChain's wrapper only writes to disk
when save_local() is called -- so this module wraps every add_documents()
call with an explicit save to keep that invisible to the rest of the app.

Everything outside this module (nodes.py, the API layer) only ever calls
get_vectorstore(), add_documents(), list_sources(), and collection_is_empty()
-- never a FAISS-specific method directly.
"""
from functools import lru_cache
from pathlib import Path

from src import config
from src.llm import get_embeddings

_FAISS_INDEX_NAME = "index"


def _get_faiss():
    from langchain_community.vectorstores import FAISS

    index_file = Path(config.VECTOR_STORE_DIR) / f"{_FAISS_INDEX_NAME}.faiss"
    embeddings = get_embeddings()

    if index_file.exists():
        return FAISS.load_local(
            config.VECTOR_STORE_DIR,
            embeddings,
            index_name=_FAISS_INDEX_NAME,
            allow_dangerous_deserialization=True,
        )

    # Bootstrap an empty FAISS index. LangChain's FAISS.from_texts() requires
    # at least one document, so we build the empty index directly instead.
    import faiss as faiss_lib
    from langchain_community.docstore.in_memory import InMemoryDocstore

    embedding_dim = len(embeddings.embed_query("dimension probe"))
    index = faiss_lib.IndexFlatL2(embedding_dim)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def _save_faiss(vs) -> None:
    Path(config.VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(config.VECTOR_STORE_DIR, index_name=_FAISS_INDEX_NAME)



# Public interface

@lru_cache(maxsize=1)
def get_vectorstore():
    return _get_faiss()


def add_documents(chunks) -> None:
    """Add chunks and persist them to disk."""
    vs = get_vectorstore()
    vs.add_documents(chunks)
    _save_faiss(vs)


def list_sources() -> list[dict]:
    """Return distinct source documents currently indexed, with chunk counts."""
    vs = get_vectorstore()
    counts: dict[str, int] = {}
    for doc_id in vs.index_to_docstore_id.values():
        doc = vs.docstore.search(doc_id)
        metadata = getattr(doc, "metadata", None) or {}
        src = metadata.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": src, "chunk_count": n} for src, n in sorted(counts.items())]


def collection_is_empty() -> bool:
    vs = get_vectorstore()
    return len(vs.index_to_docstore_id) == 0