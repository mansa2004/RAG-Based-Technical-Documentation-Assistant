# Technical Documentation RAG Assistant

A self-corrective RAG system built with **LangGraph** and served via **FastAPI**. It answers questions
over a small corpus of technical documentation, grades its own retrieved context before answering, and
retries with a rewritten query (up to a limit) when nothing relevant comes back — falling back to an
honest "I don't know" instead of hallucinating.

Built for the Express Analytics AI/ML Engineer Intern take-home assignment.

---

## 1. Overview

- **Corpus**: 5 original markdown documents covering FastAPI (basics, path/query params, request bodies,
  dependency injection, error handling) — written specifically for this assignment rather than scraped, so
  the corpus is small, controllable, and free of copyright concerns. Swap in any other docs by dropping
  `.md`/`.txt`/`.html` files into `corpus/`.
- **Workflow**: a LangGraph `StateGraph` with query analysis, retrieval, self-corrective document grading,
  generation, and an optional hallucination/groundedness check.
- **API**: FastAPI with `/query`, `/ingest`, `/documents`, `/feedback`.
- **Vector store**: ChromaDB, persisted to disk.
- **Embeddings**: local `sentence-transformers` model by default (no API key required); OpenAI embeddings
  supported as a drop-in alternative.
- **LLM**: pluggable — Groq, Google Gemini, OpenAI, or Anthropic, chosen via one environment variable.

---

## 2. Architecture

```
                     ┌──────────────────┐
        question ──► │  analyze_query   │  rewrite + classify query type
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
              ┌─────►│    retrieve      │  vector similarity search (top-k)
              │      └────────┬─────────┘
              │               ▼
              │      ┌──────────────────┐
              │      │ grade_documents  │  LLM grades each chunk relevant/irrelevant
              │      └────────┬─────────┘
              │               │
              │     ┌─────────┼─────────────┐
              │     ▼         ▼             ▼
              │  relevant   irrelevant,   irrelevant,
              │     │      retries left   retries exhausted
              │     ▼         │             ▼
              │ ┌─────────┐   │        ┌──────────┐
              │ │generate │   │        │ give_up  │──► END ("I don't know")
              │ └────┬────┘   │        └──────────┘
              │      ▼        │
              │ ┌───────────────────┐  │
              │ │check_hallucination│  │  optional (Self-RAG-style groundedness check)
              │ └────────┬──────────┘  │
              │          ▼             │
              │         END            │
              │                        ▼
              │               ┌──────────────────┐
              └───────────────┤   rewrite_query   │  new search query, retry_count += 1
                               └──────────────────┘
```

This is implemented in `src/graph/build_graph.py` using `add_conditional_edges` off the grading node's
output. The retry loop (`rewrite_query → retrieve → grade_documents → ...`) is bounded by
`MAX_RETRIES` (default 2), checked inside `grade_documents` before the conditional edge routes.

### State schema (`src/graph/state.py`)

The key design decisions, since the assignment calls this out as a core evaluation criterion:

- **`question` vs `search_query`** are kept separate. `question` is the user's original text and is never
  mutated; `search_query` is what actually gets embedded and searched, and may be rewritten 1–2 times.
  Generation always answers `question`, even after several rewrites of `search_query` — otherwise the
  final answer could drift from what the user actually asked.
- **`retry_count`** is a plain `int`, explicitly incremented by `rewrite_query`, not a LangGraph reducer.
  The retry-limit check needs to be a simple, synchronous comparison against `MAX_RETRIES` inside the
  conditional edge, so an accumulator/reducer pattern would add indirection without benefit here.
- **`documents` vs `graded_documents`** — retrieval output is kept separate from the post-grading filtered
  set, so the trace/response can show "N retrieved, M kept" for debuggability, instead of silently
  overwriting the raw retrieval.
- **`route`** is a small string set by `grade_documents` (`"generate" | "retry" | "give_up"`) and read by a
  plain Python conditional-edge function (`_route_after_grading`). Keeping the routing decision as an
  explicit state field (rather than re-deriving it inside the edge function) makes the decision testable in
  isolation — see `tests/test_graph_smoke.py`.
- **`trace`** uses LangGraph's reducer pattern (`Annotated[list[str], add]`) since it's the one field that
  should genuinely accumulate across every node in the run, for observability.

---

## 3. Setup

### Requirements
- Python 3.11+
- An API key for one LLM provider (Groq's free tier is easiest to get started with)

### Install

```bash
git clone <this-repo>
cd rag-assistant
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `LLM_PROVIDER` + the matching API key. Defaults use local embeddings
(`sentence-transformers`), so no embedding API key is required out of the box.

### Run

```bash
uvicorn src.api.main:app --reload
```

On first startup the app automatically ingests everything in `corpus/` into ChromaDB (persisted under
`data/chroma/`) if the collection is empty. Visit `http://localhost:8000/docs` for interactive Swagger UI.

### Run tests

```bash
pytest tests/ -v
```

The smoke tests cover chunking and graph-routing logic without requiring an LLM API key or network
access (they don't call `analyze_query`/`grade_documents`/`generate` directly — see `tests/test_graph_smoke.py`
for what's covered offline vs. what needs a live key).

---

## 4. Example API requests

**Ask a question**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I define a request body with validation in FastAPI?"}'
```
```json
{
  "question": "How do I define a request body with validation in FastAPI?",
  "answer": "Define a class inheriting from Pydantic's BaseModel ... [03_request_body_and_pydantic.md]",
  "sources": ["03_request_body_and_pydantic.md"],
  "query_type": "how-to",
  "search_query": "FastAPI request body validation Pydantic BaseModel",
  "retries_used": 0,
  "graded_documents": [ { "source": "03_request_body_and_pydantic.md", "chunk_id": "...", "relevant": true, "reasoning": "..." } ],
  "answer_is_grounded": true,
  "hallucination_reason": "Answer content matches the provided context.",
  "trace": ["analyze_query: ...", "retrieve: 4 chunks ...", "grade_documents: 3/4 relevant -> route=generate", "generate: ...", "check_hallucination: grounded=True"]
}
```

**Ingest a new document by URL**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "urls=https://example.com/some-doc.md"
```

**Ingest a file upload**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@/path/to/notes.md"
```

**List indexed documents**
```bash
curl http://localhost:8000/documents
```

**Submit feedback**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "answer": "...", "rating": "up", "comment": "Accurate and well cited"}'
```

---

## 5. Chunking & embedding strategy

Technical docs mix prose with code blocks and headers; naive fixed-size character splitting risks slicing a
code block in half, which destroys its meaning as retrieved context. The pipeline (`src/ingestion.py`) does
a **two-pass split**:

1. `MarkdownHeaderTextSplitter` splits on `#`/`##`/`###` first, so each resulting piece keeps its section
   header as metadata (`h1`/`h2`/`h3`). This both improves grading (the LLM sees what section a chunk is
   from) and improves citation quality.
2. `RecursiveCharacterTextSplitter` then enforces a hard size bound (`CHUNK_SIZE=700` chars,
   `CHUNK_OVERLAP=100`, ~15%) using separators ordered `["\n\`\`\`", "\n\n", "\n", ". ", " ", ""]` — code
   fences and paragraph breaks are preferred split points over mid-sentence breaks.

**Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` by default — it's free, requires no API key,
runs on CPU, and is more than adequate for a 3–5 document corpus. `OPENAI_EMBEDDING_MODEL` is a
one-line swap in `.env` if higher retrieval quality is needed at scale.

**Vector store**: pluggable via `VECTOR_STORE_PROVIDER` (`chroma` default, or `faiss`). `src/vectorstore.py`
exposes one provider-agnostic interface (`get_vectorstore`, `add_documents`, `list_sources`,
`collection_is_empty`) so `src/graph/nodes.py` and the API layer never touch a provider-specific method —
switching backends is a one-line env var change.

### 5a. Chroma vs. FAISS — comparison and how to switch

| Factor | Chroma (default) | FAISS |
|---|---|---|
| Raw ANN search speed at scale | Good | Better, especially with IVF/HNSW index types on 100k+ vectors |
| Metadata storage | Native, queryable | Via LangChain's `InMemoryDocstore` — works, but not a first-class query feature |
| Persistence | Automatic on every write | Manual — `save_local()`/`load_local()`; `src/vectorstore.py` wraps every write with an explicit save so this stays invisible to the rest of the app |
| Dynamic `POST /ingest` at runtime | Natural fit | Requires the explicit re-save above; supported here, just less native |
| Dependency weight | Heavier (its own storage engine) | Lighter, pure numerical library |
| Metadata filtering during search | Built-in `where` filters | Post-filter only |

**Why Chroma is the default here:** this project's `/ingest` endpoint writes documents at runtime and
`/documents` needs to list per-source metadata — both are things Chroma does natively. FAISS becomes the
better choice once the corpus is large (100k+ chunks) and ingestion is mostly batch/offline rather than
live via the API.

**To switch:** set `VECTOR_STORE_PROVIDER=faiss` in `.env` (`faiss-cpu` is already in `requirements.txt`).
Nothing else changes — `src/graph/nodes.py` calls `similarity_search_with_score()`, which both backends
implement identically, so the graph itself has zero branching on vector store choice. Each provider gets its
own subfolder under `data/` so switching back and forth never loads stale data from the other backend.

---

## 6. Design decisions & tradeoffs

| Decision | Reasoning | Tradeoff accepted |
|---|---|---|
| Explicit `StateGraph` with conditional edges over a ReAct-style tool-calling agent | Matches the assignment's "self-corrective" spec directly; routing is deterministic and unit-testable independent of any LLM call | Less flexible than letting an agent freely decide what to do next |
| Grade each chunk individually rather than grading the whole retrieved set at once | Lets partially-relevant retrievals through (keep the 2 good chunks, drop the 2 bad ones) instead of an all-or-nothing decision | One LLM call per retrieved chunk — more latency/cost per query than a single batched grading call |
| `retry_count` as a plain int with a hard `MAX_RETRIES` | Guarantees the graph terminates; a `while not satisfied` loop with no ceiling risks infinite loops on a genuinely out-of-corpus question | Legitimate questions requiring 3+ rewrites will still hit "I don't know" |
| SQLite for feedback instead of a flat JSON file | Safe concurrent writes without adding real infrastructure | Adds a (tiny) schema to maintain vs. just appending to a file |
| Local embeddings by default | Zero-friction setup, no embedding API key needed to try the project | Lower retrieval quality ceiling than OpenAI/Cohere embeddings on more diverse/larger corpora |
| Vector store made pluggable (Chroma default, FAISS supported) | Chroma's native metadata + auto-persistence fit this project's live `/ingest` endpoint best; a one-line env swap to FAISS covers larger-scale/offline-ingestion use cases | Small abstraction overhead (`src/vectorstore.py`) to keep both backends behind one interface |
| Hallucination check as a separate node after generation, toggleable | Keeps the core required pipeline (analysis→retrieve→grade→generate) uncluttered while still demonstrating the Self-RAG-style bonus | Doubles the LLM calls on the generation path when enabled |

---

## 7. What I'd improve with more time

- **Batch the grading calls** into a single LLM call per retrieval (one prompt listing all k chunks, asking
  for a JSON array of relevance judgments) instead of k separate calls — cuts latency and cost noticeably.
- **Streaming responses** from `/query` (SSE or chunked) so the UI can show the answer as it's generated
  rather than waiting for the full graph to finish.
- **Web search fallback** (stubbed via `ENABLE_WEB_SEARCH_FALLBACK` in config but not wired into the graph
  yet) — after `give_up`, optionally call Tavily and generate from web results instead of just refusing.
- **Conversation memory** — the state schema already carries `session_id`; the next step is persisting prior
  turns (e.g., in SQLite alongside feedback) and folding relevant history into `analyze_query`.
- **Re-ranking** retrieved chunks with a cross-encoder before grading, to cut down on how often the grader
  has to reject chunks that were only superficially similar in embedding space.
- **Async LLM/embedding calls** in the API layer — currently synchronous, which is fine for a single-user
  demo but would bottleneck under concurrent load.

## 8. Assumptions made

- The corpus is small enough (3–5 docs) that simple top-k similarity search is sufficient; no need for
  hybrid (BM25 + vector) search at this scale.
- "Grading" is done per-chunk by an LLM call rather than a cheaper heuristic (e.g., cosine similarity
  threshold), since the assignment explicitly asks for an LLM-based grader as the self-corrective
  component.
- A single retry limit (`MAX_RETRIES=2`) applies uniformly regardless of query type; a more sophisticated
  system might vary this by `query_type` (e.g., allow more retries for `troubleshooting` questions).
- File uploads for `/ingest` are restricted to `.md`, `.txt`, `.html` to match the ingestion pipeline's loaders;
  PDF ingestion was out of scope for the 2-day window.

---

## 9. Project structure

```
rag-assistant/
├── README.md
├── requirements.txt
├── .env.example
├── corpus/                  # 5 original markdown docs (FastAPI topics)
├── src/
│   ├── config.py             # all tunables, env-var driven
│   ├── llm.py                 # provider-agnostic LLM + embeddings factory
│   ├── ingestion.py           # load -> chunk -> embed -> store
│   ├── vectorstore.py         # pluggable Chroma/FAISS vector store wrapper
│   ├── graph/
│   │   ├── state.py           # GraphState TypedDict
│   │   ├── nodes.py           # analyze_query, retrieve, grade_documents, generate, rewrite_query, check_hallucination
│   │   └── build_graph.py     # StateGraph wiring + conditional edges
│   └── api/
│       ├── main.py            # FastAPI app: /query /ingest /documents /feedback
│       ├── models.py          # Pydantic request/response schemas
│       └── feedback_store.py  # SQLite feedback persistence
├── scripts/
│   └── ingest.py               # standalone CLI ingestion
└── tests/
    └── test_graph_smoke.py     # offline tests: chunking + routing logic
```
