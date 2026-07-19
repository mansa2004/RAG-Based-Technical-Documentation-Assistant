"""
Node implementations for the self-corrective RAG graph.

Each node is a plain function: (GraphState) -> partial GraphState update.
This matches LangGraph's convention of returning only the keys a node changes;
LangGraph merges the returned dict into the running state.
"""
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from src import config
from src.llm import get_llm
from src.vectorstore import get_vectorstore

# ---------------------------------------------------------------------------
# Node 1: Query Analysis
# ---------------------------------------------------------------------------
_QUERY_ANALYSIS_PROMPT = """You rewrite user questions about technical documentation to improve \
vector search retrieval, and classify the question type.

Rewrite the question to be a clearer, more specific search query: expand abbreviations, \
add likely synonyms, and resolve vague pronouns. Keep it short (1-2 sentences). Do not \
answer the question, only rewrite it for search.

Then classify the question into exactly one of: conceptual, how-to, troubleshooting, api_reference.

Respond ONLY with JSON in this exact shape, no other text:
{{"rewritten_query": "...", "query_type": "..."}}

Original question: {question}"""


def analyze_query(state: dict) -> dict:
    question = state["question"]
    llm = get_llm()

    raw = llm.invoke(
        [
            SystemMessage(content="You are a precise query-rewriting assistant. Output valid JSON only."),
            HumanMessage(content=_QUERY_ANALYSIS_PROMPT.format(question=question)),
        ]
    )
    content = _get_text(raw)

    try:
        parsed = json.loads(_extract_json(content))
        rewritten = parsed.get("rewritten_query", question) or question
        query_type = parsed.get("query_type", "conceptual")
    except Exception:
        rewritten = question
        query_type = "conceptual"

    return {
        "search_query": rewritten,
        "query_type": query_type,
        "retry_count": state.get("retry_count", 0),
        "trace": [f"analyze_query: rewritten='{rewritten}' type={query_type}"],
    }


# ---------------------------------------------------------------------------
# Node 2: Retrieval
# ---------------------------------------------------------------------------
def retrieve(state: dict) -> dict:
    query = state.get("search_query") or state["question"]
    vs = get_vectorstore()

    results = vs.similarity_search_with_score(query, k=config.TOP_K)

    documents = [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "score": float(score),
        }
        for doc, score in results
    ]

    return {
        "documents": documents,
        "trace": [f"retrieve: {len(documents)} chunks for query='{query}'"],
    }


# ---------------------------------------------------------------------------
# Node 3: Document Grading (self-corrective core)
# ---------------------------------------------------------------------------
_GRADE_PROMPT = """You are grading whether a retrieved document chunk is relevant to a user's question.

Question: {question}

Document chunk:
\"\"\"
{chunk}
\"\"\"

Is this chunk relevant enough to help answer the question? Respond ONLY with JSON:
{{"relevant": true or false, "reasoning": "one short sentence"}}"""


def grade_documents(state: dict) -> dict:
    question = state["question"]
    documents = state.get("documents", [])
    llm = get_llm()

    graded = []
    for doc in documents:
        raw = llm.invoke(
            [
                SystemMessage(content="You are a strict relevance grader. Output valid JSON only."),
                HumanMessage(
                    content=_GRADE_PROMPT.format(question=question, chunk=doc["content"][:1500])
                ),
            ]
        )
        content = _get_text(raw)
        try:
            parsed = json.loads(_extract_json(content))
            relevant = bool(parsed.get("relevant", False))
            reasoning = parsed.get("reasoning", "")
        except Exception:
            relevant, reasoning = False, "grading parse failure, treated as irrelevant"

        graded.append(
            {
                "content": doc["content"],
                "source": doc["source"],
                "chunk_id": doc["chunk_id"],
                "relevant": relevant,
                "reasoning": reasoning,
            }
        )

    kept = [g for g in graded if g["relevant"]]
    retry_count = state.get("retry_count", 0)

    if kept:
        route = "generate"
    elif retry_count < config.MAX_RETRIES:
        route = "retry"
    else:
        route = "give_up"

    return {
        "graded_documents": graded,
        "route": route,
        "trace": [f"grade_documents: {len(kept)}/{len(graded)} relevant -> route={route}"],
    }


def route_after_grading(state: dict) -> str:
    """Conditional edge function read by the graph builder."""
    return state.get("route", "give_up")


# ---------------------------------------------------------------------------
# Fallback: Rewrite query and increment retry counter
# ---------------------------------------------------------------------------
_REWRITE_PROMPT = """The following search query returned no relevant documents from a technical \
documentation vector store. Rewrite it as a substantially different query -- try different \
terminology, a broader or narrower phrasing, or a different angle on the same underlying question. \
Keep it to one sentence.

Original question: {question}
Previous search query: {search_query}

Respond ONLY with JSON: {{"rewritten_query": "..."}}"""


def rewrite_query(state: dict) -> dict:
    llm = get_llm()
    raw = llm.invoke(
        [
            SystemMessage(content="You rewrite failed search queries. Output valid JSON only."),
            HumanMessage(
                content=_REWRITE_PROMPT.format(
                    question=state["question"], search_query=state.get("search_query", "")
                )
            ),
        ]
    )
    content = _get_text(raw)
    try:
        parsed = json.loads(_extract_json(content))
        new_query = parsed.get("rewritten_query") or state["question"]
    except Exception:
        new_query = state["question"]

    new_retry_count = state.get("retry_count", 0) + 1

    return {
        "search_query": new_query,
        "retry_count": new_retry_count,
        "trace": [f"rewrite_query: attempt {new_retry_count} -> '{new_query}'"],
    }


def give_up(state: dict) -> dict:
    """Terminal fallback when no relevant docs are found after retries (and web search,
    if enabled, also came up empty)."""
    return {
        "answer": (
            "I don't have enough information in the indexed documentation to answer that "
            "confidently. Could you rephrase the question, or is it possible this topic "
            "isn't covered in the current corpus?"
        ),
        "sources": [],
        "trace": ["give_up: exhausted retries with no relevant documents"],
    }


# ---------------------------------------------------------------------------
# Bonus Node: Web search fallback
# ---------------------------------------------------------------------------
def web_search_fallback(state: dict) -> dict:
    """
    Only reached when the corpus has nothing relevant after MAX_RETRIES rewrites, and
    ENABLE_WEB_SEARCH_FALLBACK=true. Queries Tavily and repackages the results in the
    same shape as `graded_documents` (source=URL, relevant=True) so the existing
    `generate` node can consume them completely unchanged -- no branching needed there.

    Any failure here (no API key, network error, zero results) sets route="give_up" so
    the graph still terminates with the honest fallback message rather than crashing or
    generating from an empty context.
    """
    if not config.TAVILY_API_KEY:
        return {
            "route": "give_up",
            "trace": ["web_search_fallback: no TAVILY_API_KEY configured, falling back to give_up"],
        }

    query = state.get("search_query") or state["question"]

    try:
        from src.llm import get_web_search_client

        client = get_web_search_client()
        response = client.search(query=query, max_results=4)
        hits = response.get("results", [])
    except Exception as e:
        return {
            "route": "give_up",
            "trace": [f"web_search_fallback: search failed ({e}), falling back to give_up"],
        }

    if not hits:
        return {
            "route": "give_up",
            "trace": ["web_search_fallback: no web results found, falling back to give_up"],
        }

    graded = [
        {
            "content": (hit.get("content") or "")[:2000],
            "source": hit.get("url", "web"),
            "chunk_id": hit.get("url", "web"),
            "relevant": True,
            "reasoning": "sourced from live web search fallback, not the indexed corpus",
        }
        for hit in hits
    ]

    return {
        "graded_documents": graded,
        "route": "generate",
        "used_web_fallback": True,
        "trace": [f"web_search_fallback: {len(graded)} web result(s) retrieved for query='{query}'"],
    }


def route_after_web_search(state: dict) -> str:
    """Conditional edge function read by the graph builder after web_search_fallback."""
    return "generate" if state.get("route") == "generate" else "give_up"


# ---------------------------------------------------------------------------
# Node 4: Generation
# ---------------------------------------------------------------------------
_GENERATE_PROMPT = """Answer the user's question using ONLY the provided context. If the context does not \
fully answer the question, say what is missing rather than guessing.

Cite sources inline using the format [source] right after the sentence(s) it supports.

Question: {question}

Context:
{context}

Write a clear, accurate answer with inline [source] citations."""


def generate(state: dict) -> dict:
    question = state["question"]
    graded = [g for g in state.get("graded_documents", []) if g["relevant"]]

    context = "\n\n".join(
        f"[{g['source']}]\n{g['content']}" for g in graded
    )

    llm = get_llm()
    raw = llm.invoke(
        [
            SystemMessage(content="You are a precise technical documentation assistant."),
            HumanMessage(content=_GENERATE_PROMPT.format(question=question, context=context)),
        ]
    )
    answer = _get_text(raw)

    if state.get("used_web_fallback"):
        answer += (
            "\n\n*Note: the indexed documentation had nothing relevant, so this answer is "
            "based on a live web search instead.*"
        )

    sources = sorted({g["source"] for g in graded})

    return {
        "answer": answer,
        "sources": sources,
        "trace": [f"generate: answer produced from {len(graded)} chunks, {len(sources)} sources"],
    }


# ---------------------------------------------------------------------------
# Bonus Node: Hallucination / groundedness check (Self-RAG-inspired)
# ---------------------------------------------------------------------------
_GROUNDEDNESS_PROMPT = """Does the ANSWER below rely only on facts present in the CONTEXT, or does it \
introduce claims not supported by the context?

CONTEXT:
{context}

ANSWER:
{answer}

Respond ONLY with JSON: {{"grounded": true or false, "reason": "one short sentence"}}"""


def check_hallucination(state: dict) -> dict:
    graded = [g for g in state.get("graded_documents", []) if g["relevant"]]
    context = "\n\n".join(g["content"] for g in graded)
    answer = state.get("answer", "")

    llm = get_llm()
    raw = llm.invoke(
        [
            SystemMessage(content="You are a strict groundedness auditor. Output valid JSON only."),
            HumanMessage(content=_GROUNDEDNESS_PROMPT.format(context=context, answer=answer)),
        ]
    )
    content = _get_text(raw)
    try:
        parsed = json.loads(_extract_json(content))
        grounded = bool(parsed.get("grounded", True))
        reason = parsed.get("reason", "")
    except Exception:
        grounded, reason = True, "groundedness parse failure, defaulting to grounded"

    update = {
        "answer_is_grounded": grounded,
        "hallucination_reason": reason,
        "trace": [f"check_hallucination: grounded={grounded} ({reason})"],
    }

    if not grounded:
        update["answer"] = (
            state.get("answer", "")
            + "\n\n*Note: this answer may include claims not fully supported by the "
            "retrieved documentation. Treat it with extra caution.*"
        )

    return update


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _get_text(raw) -> str:
    """
    Extract plain text from an LLM response. LangChain messages type `.content` as
    `str | list[str | dict]` (to support multimodal content blocks), which is why a bare
    `raw.content` doesn't satisfy a `str` parameter for a type checker -- and isn't
    actually guaranteed to be a plain string at runtime either. All our prompts are
    text-only, so in practice this always returns a string, but this handles the list
    case properly instead of just silencing the type checker.
    """
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> str:
    """LLMs sometimes wrap JSON in markdown fences or add stray text; extract the {...} span."""
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return text[start : end + 1]