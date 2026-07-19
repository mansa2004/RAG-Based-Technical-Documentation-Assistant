"""
Vector store abstraction supporting two interchangeable backends, selected via
VECTOR_STORE_PROVIDER in .env:

  - "chroma" (default): native metadata storage + query-time filtering, and
    persists to disk automatically on every write. Best fit when documents are
    ingested dynamically at runtime via POST /ingest, since nothing extra has
    to be done to make a new write durable.

  - "faiss": Meta's similarity-search library. Faster raw nearest-neighbor
    search at large scale (with IVF/HNSW index types), but persistence is
    manual -- LangChain's FAISS wrapper only writes to disk when save_local()
    is called, so this module wraps every add_documents() call with an
    explicit save to keep that invisible to the rest of the app.

Everything outside this module (nodes.py, the API layer) only ever calls
get_vectorstore(), add_documents(), list_sources(), and collection_is_empty()
-- never a provider-specific method directly -- which is what makes the
provider swappable via one env var with zero changes anywhere else. Both
backends implement LangChain's standard similarity_search_with_score()
identically, so src/graph/nodes.py needs no branching at all.
"""
from functools import lru_cache
from pathlib import Path

from src import config
from src.llm import get_embeddings

_FAISS_INDEX_NAME = "index"


# ---------------------------------------------------------------------------
# Chroma backend
# ---------------------------------------------------------------------------
def _get_chroma():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=config.VECTOR_STORE_DIR,
    )


def _list_sources_chroma(vs) -> list[dict]:
    data = vs.get(include=["metadatas"])
    metadatas = data.get("metadatas", []) or []
    counts: dict[str, int] = {}
    for m in metadatas:
        src = (m or {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": src, "chunk_count": n} for src, n in sorted(counts.items())]


def _is_empty_chroma(vs) -> bool:
    return vs._collection.count() == 0


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------
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


def _list_sources_faiss(vs) -> list[dict]:
    counts: dict[str, int] = {}
    for doc in vs.docstore._dict.values():
        src = (doc.metadata or {}).get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return [{"source": src, "chunk_count": n} for src, n in sorted(counts.items())]


def _is_empty_faiss(vs) -> bool:
    return len(vs.docstore._dict) == 0


# ---------------------------------------------------------------------------
# Public interface (provider-agnostic)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_vectorstore():
    provider = config.VECTOR_STORE_PROVIDER.lower()
    if provider == "chroma":
        return _get_chroma()
    if provider == "faiss":
        return _get_faiss()
    raise ValueError(f"Unknown VECTOR_STORE_PROVIDER '{provider}'. Expected 'chroma' or 'faiss'.")


def add_documents(chunks) -> None:
    """Add chunks and guarantee they are durable, regardless of backend."""
    vs = get_vectorstore()
    vs.add_documents(chunks)
    if config.VECTOR_STORE_PROVIDER.lower() == "faiss":
        _save_faiss(vs)
    # Chroma persists automatically on write -- nothing further needed.


def list_sources() -> list[dict]:
    """Return distinct source documents currently indexed, with chunk counts."""
    vs = get_vectorstore()
    provider = config.VECTOR_STORE_PROVIDER.lower()
    if provider == "faiss":
        return _list_sources_faiss(vs)
    return _list_sources_chroma(vs)


def collection_is_empty() -> bool:
    vs = get_vectorstore()
    provider = config.VECTOR_STORE_PROVIDER.lower()
    if provider == "faiss":
        return _is_empty_faiss(vs)
    return _is_empty_chroma(vs)
