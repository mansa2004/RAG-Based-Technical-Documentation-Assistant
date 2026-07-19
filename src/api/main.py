import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src import config
from src.graph.build_graph import run_query
from src.ingestion import ingest_corpus_dir, ingest_paths
from src.vectorstore import list_sources, collection_is_empty
from src.api.feedback_store import save_feedback
from src.api.models import (
    QueryRequest,
    QueryResponse,
    GradedDocOut,
    IngestUrlsRequest,
    IngestResponse,
    DocumentsResponse,
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-assistant")

app = FastAPI(
    title="Technical Documentation RAG Assistant",
    description=(
        "Self-corrective RAG over a technical documentation corpus, built with a "
        "LangGraph workflow (query analysis -> retrieval -> grading -> generation)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_ingest() -> None:
    """Ingest the bundled corpus on first run if the vector store is empty."""
    try:
        if collection_is_empty():
            n = ingest_corpus_dir()
            logger.info("Startup ingestion: added %d chunks from %s", n, config.CORPUS_DIR)
        else:
            logger.info("Vector store already populated, skipping startup ingestion.")
    except Exception as e:
        # Don't crash the app if the corpus dir is missing/empty; /ingest can be used later.
        logger.warning("Startup ingestion skipped due to error: %s", e)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    """
    Landing page. Only GET endpoints can be real clickable links (POST endpoints like
    /query, /ingest, /feedback need a JSON body, which a plain link can't send) -- those
    are routed to /docs instead, where Swagger provides a form to fill in and submit them.
    """
    return """
    <html>
      <head><title>RAG Assistant</title></head>
      <body style="font-family: sans-serif; max-width: 640px; margin: 60px auto; line-height: 1.6;">
        <h2>Technical Documentation RAG Assistant</h2>
        <p>Self-corrective RAG over a technical documentation corpus, built with LangGraph + FastAPI.</p>
        <h3>Try it</h3>
        <ul>
          <li><a href="/docs">/docs</a> &mdash; interactive Swagger UI (use this for POST /query, /ingest, /feedback)</li>
          <li><a href="/redoc">/redoc</a> &mdash; alternative API reference</li>
        </ul>
        <h3>Quick GET links</h3>
        <ul>
          <li><a href="/health">/health</a> &mdash; service status</li>
          <li><a href="/documents">/documents</a> &mdash; list indexed documents</li>
        </ul>
      </body>
    </html>
    """


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if collection_is_empty():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No documents indexed yet. Call POST /ingest first.",
        )

    try:
        result = run_query(req.question, session_id=req.session_id)
    except Exception as e:
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    graded = [
        GradedDocOut(
            source=g["source"], chunk_id=g["chunk_id"], relevant=g["relevant"], reasoning=g["reasoning"]
        )
        for g in result.get("graded_documents", [])
    ]
    used_web_fallback=result.get("used_web_fallback", False),

    return QueryResponse(
        question=req.question,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        query_type=result.get("query_type"),
        search_query=result.get("search_query"),
        retries_used=result.get("retry_count", 0),
        graded_documents=graded,
        answer_is_grounded=result.get("answer_is_grounded"),
        hallucination_reason=result.get("hallucination_reason"),
        trace=result.get("trace", []),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(urls: IngestUrlsRequest = IngestUrlsRequest(), files: list[UploadFile] = File(default=[])):
    if not urls.urls and not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one file upload or one URL to ingest.",
        )

    tmp_paths: list[str] = []
    try:
        if files:
            tmp_dir = tempfile.mkdtemp()
            for f in files:
                if not f.filename:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Uploaded file is missing a filename.",
                    )
                filename: str = f.filename
                suffix = Path(filename).suffix.lower()
                if suffix not in (".md", ".txt", ".html"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported file type '{suffix}'. Allowed: .md, .txt, .html",
                    )
                dest = Path(tmp_dir) / filename
                with open(dest, "wb") as out:
                    shutil.copyfileobj(f.file, out)
                tmp_paths.append(str(dest))

        added = ingest_paths(paths=tmp_paths, urls=[str(u) for u in urls.urls])
        return IngestResponse(chunks_added=added, message=f"Indexed {added} chunks.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    finally:
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)


@app.get("/documents", response_model=DocumentsResponse)
def documents():
    sources = list_sources()
    return DocumentsResponse(
        total_sources=len(sources),
        total_chunks=sum(s["chunk_count"] for s in sources),
        documents=[DocumentInfo(**s) for s in sources],
    )


@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def feedback(req: FeedbackRequest):
    try:
        fid = save_feedback(req.question, req.answer, req.rating, req.comment, req.session_id)
    except Exception as e:
        logger.exception("Saving feedback failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return FeedbackResponse(id=fid, message="Feedback recorded, thank you.")