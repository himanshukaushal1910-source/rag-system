from __future__ import annotations

import time
import uuid
from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api.exceptions import NotFoundError
from api.schemas import (
    HealthResponse,
    IngestJobResponse,
    IngestRequest,
    IngestStatusResponse,
)
from config import get_settings
from ingestion.ingestor import ingest_directory, ingest_pdf
from retrieval.qdrant_client import QdrantClientSingleton, ensure_collection_exists

router = APIRouter(tags=["ingestion"])
logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Job store with timestamps for TTL eviction (L-3)
_jobs: dict[str, dict] = {}
_JOB_TTL_SECONDS = 86_400  # 24 hours

_MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB (M-1)


def _evict_old_jobs() -> None:
    """Purge completed/failed jobs older than _JOB_TTL_SECONDS."""
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale = [jid for jid, job in _jobs.items() if job.get("created_at", 0) < cutoff]
    for jid in stale:
        del _jobs[jid]


def _upload_dir() -> Path:
    """Return configured upload directory (never hardcoded)."""
    return Path(get_settings().upload_dir)


def _resolve_safe_upload_path(filename: str) -> Path | None:
    """Return a safe save path inside upload_dir, or None if traversal detected.

    Strips directory components and validates the resolved path stays inside
    the upload directory (C-2 fix).
    """
    # Keep only the basename — rejects any path separator
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename.replace("\\", "/").split("/")[-1]:
        return None
    upload_dir = _upload_dir().resolve()
    candidate = (upload_dir / safe_name).resolve()
    # Ensure the resolved path is strictly inside upload_dir
    try:
        candidate.relative_to(upload_dir)
    except ValueError:
        return None
    return candidate


def _validate_ingest_directory(directory_str: str) -> Path | None:
    """Validate that the ingest directory is within an allowed root (C-3).

    If allowed_ingest_roots is empty, only the configured upload_dir is allowed.
    Returns the resolved Path on success, None on violation.
    """
    settings = get_settings()
    resolved = Path(directory_str).resolve()

    allowed_roots = settings.allowed_ingest_roots
    if allowed_roots:
        for root in allowed_roots:
            try:
                resolved.relative_to(Path(root).resolve())
                return resolved
            except ValueError:
                continue
        return None

    # Default: only allow the configured upload_dir
    upload_dir = Path(settings.upload_dir).resolve()
    try:
        resolved.relative_to(upload_dir)
        return resolved
    except ValueError:
        return None


async def _run_ingestion_job(job_id: str, req: IngestRequest, app_state: object) -> None:
    """Background task for directory/filename ingestion."""
    log = logger.bind(job_id=job_id)
    _jobs[job_id]["status"] = "running"
    log.info("Ingestion job started")
    try:
        directory = _validate_ingest_directory(req.directory)
        if directory is None:
            raise ValueError(
                "Requested directory is outside allowed ingestion roots. "
                "Update ALLOWED_INGEST_ROOTS in your .env to permit it."
            )

        # Reuse app-level components (L-10)
        embedder = getattr(app_state, "embedder", None)
        sparse_encoder = getattr(app_state, "sparse_encoder", None)

        if req.filename:
            pdf_path = directory / req.filename
            result = await ingest_pdf(
                pdf_path,
                doc_id=req.doc_id,
                embedder=embedder,
                sparse_encoder=sparse_encoder,
            )
            results = [result]
        else:
            results = await ingest_directory(directory)
        _jobs[job_id].update({"status": "completed", "result": results})
        log.info("Ingestion job completed", files=len(results))
    except Exception as exc:
        log.error("Ingestion job failed", error=str(exc))
        _jobs[job_id].update({"status": "failed", "error": str(exc)})


@router.post("/api/ingest/file")
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
) -> JSONResponse:
    """Upload a PDF from browser and ingest it immediately.

    Security:
      - Only .pdf files accepted
      - Filename sanitised — no path traversal (C-2)
      - File size capped at 50 MB (M-1)
      - Reuses app-level embedder/sparse_encoder (L-10)
    """
    log = logger.bind(filename=file.filename)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=422,
            content={"error": "Only PDF files are supported."},
        )

    # ── Sanitise filename — reject path traversal (C-2) ──────────────────
    save_path = _resolve_safe_upload_path(file.filename)
    if save_path is None:
        return JSONResponse(
            status_code=422,
            content={"error": "Invalid filename."},
        )

    # ── Enforce file size limit (M-1) ─────────────────────────────────────
    content = await file.read(_MAX_UPLOAD_BYTES + 1)
    if len(content) > _MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": f"File exceeds {_MAX_UPLOAD_BYTES // (1024*1024)} MB limit."},
        )

    _upload_dir().mkdir(parents=True, exist_ok=True)

    try:
        save_path.write_bytes(content)
        log.info("PDF saved", path=str(save_path))
    except Exception as exc:
        log.error("Failed to save uploaded file", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to save file."},
        )

    # Reuse app-level components (L-10)
    embedder = getattr(request.app.state, "embedder", None)
    sparse_encoder = getattr(request.app.state, "sparse_encoder", None)

    try:
        await ensure_collection_exists()
        result = await ingest_pdf(save_path, embedder=embedder, sparse_encoder=sparse_encoder)
        chunks = result.get("chunks_ingested", 0)
        skipped = result.get("pages_skipped", 0)
        log.info("File ingestion complete", chunks=chunks, skipped=skipped)
        return JSONResponse(content={
            "status": "ok",
            "filename": save_path.name,
            "chunks_ingested": chunks,
            "pages_skipped": skipped,
        })
    except Exception as exc:
        log.error("Ingestion failed", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "Ingestion failed."},
        )


@router.get("/api/files/{filename}", response_model=None)
async def serve_file(filename: str) -> FileResponse | JSONResponse:
    """Serve a PDF file for viewing in browser.

    Security: validates that the resolved path stays inside upload_dir (C-1).
    """
    upload_dir = _upload_dir().resolve()
    # Compute candidate path and resolve it before any existence check
    candidate = (upload_dir / Path(filename).name).resolve()

    # Reject traversal: resolved path must be strictly inside upload_dir
    try:
        candidate.relative_to(upload_dir)
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})

    if not candidate.exists() or not candidate.is_file():
        return JSONResponse(
            status_code=404,
            content={"error": "File not found."},
        )

    return FileResponse(
        path=str(candidate),
        media_type="application/pdf",
    )


@router.post("/api/ingest", response_model=IngestJobResponse)
async def ingest(
    request: Request,
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestJobResponse:
    """Trigger directory ingestion as a background job."""
    _evict_old_jobs()

    # Validate directory before queuing (C-3)
    if _validate_ingest_directory(body.directory) is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Directory is outside allowed ingestion roots.",
                "detail": "Configure ALLOWED_INGEST_ROOTS in your .env to permit additional paths.",
            },
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "queued",
        "result": None,
        "error": None,
        "created_at": time.time(),
    }
    background_tasks.add_task(_run_ingestion_job, job_id, body, request.app.state)
    logger.info("Ingestion job queued", job_id=job_id, directory=body.directory)
    return IngestJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion started. Poll /api/ingest/{job_id} for status.",
    )


@router.get("/api/ingest/{job_id}", response_model=IngestStatusResponse)
async def ingest_status(job_id: str) -> IngestStatusResponse:
    """Poll ingestion job status."""
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
    """Health check — Qdrant connection and chunk count."""
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
