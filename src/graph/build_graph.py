"""
Assembles the StateGraph:

    analyze_query -> retrieve -> grade_documents --(relevant)--> generate --> [hallucination_check] --> END
                                       |
                                       +--(irrelevant, retries left)--> rewrite_query -> retrieve  (loop)
                                       |
                                       +--(irrelevant, retries exhausted)-->
                                             if ENABLE_WEB_SEARCH_FALLBACK:
                                                 web_search_fallback --(results found)--> generate
                                                                     --(no results/no key)--> give_up --> END
                                             else:
                                                 give_up --> END

The retry loop is bounded by MAX_RETRIES via state["retry_count"], checked inside
grade_documents/route_after_grading -- this is what prevents an infinite loop. The web
search fallback is a second, independent safety net that only fires after the corpus
retry loop is already exhausted, and itself always terminates in exactly one hop
(no retries on the web search side) to keep the graph's termination guarantee simple.
"""
from functools import lru_cache
from typing import Hashable

from langgraph.graph import StateGraph, END

from src import config
from src.graph.state import GraphState
from src.graph import nodes


def _route_after_grading(state: dict) -> str:
    route = state.get("route", "give_up")
    if route == "generate":
        return "generate"
    if route == "retry":
        return "rewrite_query"
    # route == "give_up": send to the web search fallback node if enabled, else terminate
    if config.ENABLE_WEB_SEARCH_FALLBACK:
        return "web_search_fallback"
    return "give_up"


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("analyze_query", nodes.analyze_query)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("generate", nodes.generate)
    graph.add_node("give_up", nodes.give_up)

    if config.ENABLE_WEB_SEARCH_FALLBACK:
        graph.add_node("web_search_fallback", nodes.web_search_fallback)

    if config.ENABLE_HALLUCINATION_CHECK:
        graph.add_node("check_hallucination", nodes.check_hallucination)

    graph.set_entry_point("analyze_query")
    graph.add_edge("analyze_query", "retrieve")
    graph.add_edge("retrieve", "grade_documents")

    routing_map: dict[Hashable, str] = {"generate": "generate", "rewrite_query": "rewrite_query"}
    routing_map["web_search_fallback" if config.ENABLE_WEB_SEARCH_FALLBACK else "give_up"] = (
        "web_search_fallback" if config.ENABLE_WEB_SEARCH_FALLBACK else "give_up"
    )
    graph.add_conditional_edges("grade_documents", _route_after_grading, routing_map)

    # Loop back to retrieval after a query rewrite
    graph.add_edge("rewrite_query", "retrieve")

    if config.ENABLE_WEB_SEARCH_FALLBACK:
        graph.add_conditional_edges(
            "web_search_fallback",
            nodes.route_after_web_search,
            {"generate": "generate", "give_up": "give_up"},
        )

    if config.ENABLE_HALLUCINATION_CHECK:
        graph.add_edge("generate", "check_hallucination")
        graph.add_edge("check_hallucination", END)
    else:
        graph.add_edge("generate", END)

    graph.add_edge("give_up", END)

    return graph.compile()


def run_query(question: str, session_id: str = "default") -> dict:
    app = build_graph()
    initial_state = {
        "question": question,
        "session_id": session_id,
        "retry_count": 0,
        "trace": [],
    }
    final_state = app.invoke(initial_state)
    return final_state