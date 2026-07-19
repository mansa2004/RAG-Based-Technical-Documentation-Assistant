from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question")
    session_id: str = Field(default="default", description="Session id for future memory support")


class GradedDocOut(BaseModel):
    source: str
    chunk_id: str
    relevant: bool
    reasoning: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    query_type: Optional[str] = None
    search_query: Optional[str] = None
    retries_used: int = 0
    graded_documents: list[GradedDocOut] = []
    answer_is_grounded: Optional[bool] = None
    hallucination_reason: Optional[str] = None
    trace: list[str] = []
    used_web_fallback: bool = False


class IngestUrlsRequest(BaseModel):
    urls: list[HttpUrl] = Field(default_factory=list)


class IngestResponse(BaseModel):
    chunks_added: int
    message: str


class DocumentInfo(BaseModel):
    source: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    total_sources: int
    total_chunks: int
    documents: list[DocumentInfo]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str = Field(..., pattern="^(up|down)$")
    comment: Optional[str] = None
    session_id: str = "default"


class FeedbackResponse(BaseModel):
    id: int
    message: str
