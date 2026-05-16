from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class QueryRequest(BaseModel):
    """POST /query request body.

    Attributes:
        query: Natural language question to answer.
        doc_ids: Optional list of document UUIDs to restrict retrieval to.
        content_types: Optional list of content modalities to filter by.
        top_k: Override default top-k after reranking.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=3, max_length=2000, description="Question to answer")
    doc_ids: list[str] | None = Field(default=None, description="Filter by document UUIDs")
    content_types: list[str] | None = Field(default=None, description="Filter by content type")
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override top-k reranking")

    @field_validator("content_types")
    @classmethod
    def _validate_content_types(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        allowed = {"text", "table", "image", "chart"}
        invalid = set(v) - allowed
        if invalid:
            raise ValueError(f"Invalid content_types: {invalid}. Allowed: {allowed}")
        return v


class IngestRequest(BaseModel):
    """POST /ingest request body.

    Attributes:
        directory: Absolute path to a folder containing PDFs.
        doc_id: Optional fixed doc_id (for single-file re-ingestion).
        filename: Optional single filename within the directory.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    directory: str = Field(..., description="Absolute path to PDF directory")
    doc_id: str | None = Field(default=None, description="Fixed doc_id for re-ingestion")
    filename: str | None = Field(default=None, description="Single filename to ingest")


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class CitationResponse(BaseModel):
    """Structured citation in the API response."""

    filename: str
    page: int
    chunk_id: str


class QueryResponse(BaseModel):
    """POST /query response body."""

    answer: str
    citations: list[CitationResponse]
    faithfulness_score: float
    consistency_passed: bool
    sub_queries: list[str]
    request_id: str


class IngestJobResponse(BaseModel):
    """POST /ingest response body — returns immediately with a job_id."""

    job_id: str
    status: str = "queued"
    message: str


class IngestStatusResponse(BaseModel):
    """GET /ingest/{job_id} response body."""

    job_id: str
    status: str  # queued | running | completed | failed
    result: list[dict] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """GET /health response body."""

    status: str = "ok"
    qdrant_connected: bool
    collection: str
    chunk_count: int


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error_code: str
    message: str
    detail: str
    request_id: str | None = None
