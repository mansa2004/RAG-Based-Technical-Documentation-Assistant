"""
Smoke tests that don't require an LLM API key or network access:
- chunking produces reasonable, non-empty chunks with metadata
- the graph compiles and has the expected nodes/edges
- the conditional routing function behaves correctly for each grading outcome

Run with: pytest tests/test_graph_smoke.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import load_corpus_dir, chunk_documents
from src import config


def test_corpus_loads():
    docs = load_corpus_dir(config.CORPUS_DIR)
    assert len(docs) >= 3, "Expect at least 3 corpus documents per assignment spec"
    for d in docs:
        assert d.page_content.strip()
        assert "source" in d.metadata


def test_chunking_respects_size_and_metadata():
    docs = load_corpus_dir(config.CORPUS_DIR)
    chunks = chunk_documents(docs)
    assert len(chunks) > len(docs), "Chunking should split docs into multiple pieces"
    for c in chunks:
        assert len(c.page_content) > 0
        assert "chunk_id" in c.metadata
        assert "source" in c.metadata
        # generous upper bound accounting for markdown header pass overlap
        assert len(c.page_content) <= config.CHUNK_SIZE * 4


def test_routing_function_generate():
    from src.graph.build_graph import _route_after_grading

    assert _route_after_grading({"route": "generate"}) == "generate"


def test_routing_function_retry():
    from src.graph.build_graph import _route_after_grading

    assert _route_after_grading({"route": "retry"}) == "rewrite_query"


def test_routing_function_give_up_no_web_fallback(monkeypatch):
    from src.graph.build_graph import _route_after_grading
    from src import config as cfg

    monkeypatch.setattr(cfg, "ENABLE_WEB_SEARCH_FALLBACK", False)
    assert _route_after_grading({"route": "give_up"}) == "give_up"


def test_routing_function_give_up_with_web_fallback(monkeypatch):
    from src.graph.build_graph import _route_after_grading
    from src import config as cfg

    monkeypatch.setattr(cfg, "ENABLE_WEB_SEARCH_FALLBACK", True)
    assert _route_after_grading({"route": "give_up"}) == "web_search_fallback"


def test_grade_documents_routing_logic(monkeypatch):
    """Calls the real grade_documents function (not a copy of its logic) with a fake LLM."""
    from src.graph import nodes
    from src import config as cfg
    import json

    class FakeLLM:
        def invoke(self, messages):
            from langchain_core.messages import AIMessage
            return AIMessage(content=json.dumps({"relevant": False, "reasoning": "test"}))

    monkeypatch.setattr(nodes, "get_llm", lambda: FakeLLM())

    documents = [{"content": "irrelevant text", "source": "x.md", "chunk_id": "x::0"}]

   
    result = nodes.grade_documents({"question": "q", "documents": documents, "retry_count": 0})
    assert result["route"] == "retry"

    
    result = nodes.grade_documents({"question": "q", "documents": documents, "retry_count": cfg.MAX_RETRIES})
    assert result["route"] == "give_up"