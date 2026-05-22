"""
api/schemas.py

Pydantic v2 request/response models for all API endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #

class QueryRequest(BaseModel):
    """POST /query and POST /query/stream request body."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="Natural language question to answer.",
    )
    doc_ids: list[str] | None = Field(
        default=None,
        description="Optional list of document UUIDs to restrict retrieval to.",
    )
    content_types: list[str] | None = Field(
        default=None,
        description="Optional content type filter: text, table, image, chart.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=40,
        description="Override default top-k after reranking (max 40).",
    )

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
    """POST /ingest request body."""

    model_config = ConfigDict(str_strip_whitespace=True)

    directory: str = Field(..., description="Absolute path to PDF directory.")
    doc_id: str | None = Field(default=None, description="Fixed doc_id for re-ingestion.")
    filename: str | None = Field(default=None, description="Single filename to ingest.")

    @field_validator("filename")
    @classmethod
    def _no_path_separators_in_filename(cls, v: str | None) -> str | None:
        """Reject filenames that contain directory traversal sequences (C-3)."""
        if v is None:
            return v
        from pathlib import Path
        safe = Path(v).name
        if safe != v or ".." in v:
            raise ValueError("filename must be a plain filename with no path separators")
        return v


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #

class CitationResponse(BaseModel):
    """Structured citation returned in query responses."""

    filename: str
    page: int
    chunk_id: str


class TableResponse(BaseModel):
    """A markdown table extracted from the answer."""

    markdown: str
    caption: str = ""


class ImageResponse(BaseModel):
    """An image/chart chunk referenced in the answer."""

    filename: str
    page: int
    caption: str
    image_b64: str | None = None


class QueryResponse(BaseModel):
    """POST /query response body."""

    answer: str
    citations: list[CitationResponse]
    faithfulness_score: float
    consistency_passed: bool
    sub_queries: list[str]
    request_id: str
    tables: list[TableResponse] = Field(default_factory=list)
    images: list[ImageResponse] = Field(default_factory=list)
    completeness_score: float = Field(default=1.0)
    query_type: str = Field(default="analytical", description="Detected query type for routing.")


class IngestJobResponse(BaseModel):
    """POST /ingest response — returns immediately with job_id."""

    job_id: str
    status: str = "queued"
    message: str


class IngestStatusResponse(BaseModel):
    """GET /ingest/{job_id} response."""

    job_id: str
    status: str   # queued | running | completed | failed
    result: list[dict] | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "ok"
    qdrant_connected: bool
    collection: str
    chunk_count: int


class ErrorResponse(BaseModel):
    """Standard error response."""

    error_code: str
    message: str
    detail: str
    request_id: str | None = None
