"""
Graph state schema.

Design notes (this is the "core evaluation criterion" the assignment calls out):

- `question` is preserved unchanged as the original user question, separate from
  `search_query`, which is what actually gets embedded and sent to the retriever.
  Keeping these separate matters because generation should always answer the
  user's real question, even after the search_query has been rewritten one or
  two times to chase better retrieval results.

- `retry_count` is an int that nodes increment explicitly (not a LangGraph
  reducer/accumulator) because we need exact control over the retry limit
  check in the conditional edge -- it must be readable and comparable to
  MAX_RETRIES synchronously, with no ambiguity about ordering.

- `documents` holds ALL retrieved chunks; `graded_documents` holds only the
  ones the grading node kept. Keeping both (instead of overwriting) makes the
  graph's decisions inspectable/debuggable and lets the API surface
  "N retrieved, M kept" in responses.

- `route` is a small string field set by the grading node and read by the
  conditional edge function. LangGraph conditional edges are plain Python
  functions over state, so this keeps the routing decision explicit and
  testable independently of the LLM call that produced it.

- `answer_is_grounded` / `hallucination_reason` support the optional
  hallucination-check node (Self-RAG-style) without requiring it -- if that
  node is disabled, these fields simply stay at their defaults.
"""
from typing import Annotated, TypedDict
from operator import add


class GradedDoc(TypedDict):
    content: str
    source: str
    chunk_id: str
    relevant: bool
    reasoning: str


class GraphState(TypedDict, total=False):
    # Input
    question: str  # original, never mutated
    session_id: str

    # Query analysis
    search_query: str  # possibly rewritten version used for retrieval
    query_type: str  # conceptual | how-to | troubleshooting | api_reference

    # Retrieval
    documents: list[dict]  # all raw retrieved chunks (content, source, chunk_id, score)

    # Grading (self-corrective core)
    graded_documents: list[GradedDoc]
    route: str  # "generate" | "retry" | "give_up"
    retry_count: int

    # Generation
    answer: str
    sources: list[str]

    # Bonus: hallucination check
    answer_is_grounded: bool
    hallucination_reason: str

    # Bonus: web search fallback
    used_web_fallback: bool

    # Trace, useful for the API response / debugging (append-only across nodes)
    trace: Annotated[list[str], add]
