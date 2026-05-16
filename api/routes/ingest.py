from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Request

from api.exceptions import NotFoundError
from api.schemas import (
    HealthResponse,
    IngestJobResponse,
    IngestRequest,
    IngestStatusResponse,
)
from config import get_settings
from ingestion.ingestor import ingest_directory, ingest_pdf
from retrieval.qdrant_client import QdrantClientSingleton

router = APIRouter(tags=["ingestion"])
logger: structlog.BoundLogger = structlog.get_logger(__name__)

# In-memory job store — replace with Redis/DB for production.
_jobs: dict[str, dict] = {}


async def _run_ingestion_job(job_id: str, req: IngestRequest) -> None:
    """Background task that runs the ingestion pipeline.

    Updates the in-memory job store with status and results.

    Args:
        job_id: Unique job identifier.
        req: Ingestion request parameters.
    """
    log = logger.bind(job_id=job_id)
    _jobs[job_id]["status"] = "running"
    log.info("Ingestion job started")

    try:
        directory = Path(req.directory)

        if req.filename:
            pdf_path = directory / req.filename
            result = await ingest_pdf(pdf_path, doc_id=req.doc_id)
            results = [result]
        else:
            results = await ingest_directory(directory)

        _jobs[job_id].update({"status": "completed", "result": results})
        log.info("Ingestion job completed", files=len(results))

    except Exception as exc:
        log.error("Ingestion job failed", error=str(exc))
        _jobs[job_id].update({"status": "failed", "error": str(exc)})


@router.post("/api/ingest", response_model=IngestJobResponse)
async def ingest(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestJobResponse:
    """Trigger PDF ingestion as a background job.

    Returns immediately with a ``job_id``. Poll
    ``GET /api/ingest/{job_id}`` for status.

    Args:
        request: FastAPI request.
        body: Ingestion request with directory path.
        background_tasks: FastAPI background task runner.

    Returns:
        :class:`IngestJobResponse` with job_id.
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "queued", "result": None, "error": None}

    background_tasks.add_task(_run_ingestion_job, job_id, body)

    logger.info(
        "Ingestion job queued",
        job_id=job_id,
        directory=body.directory,
    )

    return IngestJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion started. Poll /api/ingest/{job_id} for status.",
    )


@router.get("/api/ingest/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(job_id: str) -> IngestStatusResponse:
    """Poll ingestion job status.

    Args:
        job_id: Job ID returned by POST /api/ingest.

    Returns:
        :class:`IngestStatusResponse` with current status and results.

    Raises:
        NotFoundError: If the job_id does not exist.
    """
    job = _jobs.get(job_id)
    if not job:
        raise NotFoundError(f"Job not found: {job_id}")

    return IngestStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Qdrant connection status and collection chunk count.
    """
    settings = get_settings()
    try:
        client = await QdrantClientSingleton.get()
        info = await client.get_collection(settings.qdrant_collection_name)
        chunk_count = info.points_count or 0
        connected = True
    except Exception:
        chunk_count = 0
        connected = False

    return HealthResponse(
        status="ok" if connected else "degraded",
        qdrant_connected=connected,
        collection=settings.qdrant_collection_name,
        chunk_count=chunk_count,
    )
