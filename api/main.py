from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.nodes.retriever import init_retrieval_components
from api.exceptions import RagException
from api.middleware import APIKeyMiddleware, RequestIDMiddleware
from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router
from config import get_settings
from ingestion.embedder import AsyncEmbedder
from ingestion.sparse_encoder import SparseEncoder
from retrieval.qdrant_client import QdrantClientSingleton, ensure_collection_exists
from retrieval.reranker import CrossEncoderReranker

logger: structlog.BoundLogger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan: initialise heavy resources once at startup.

    Startup:
    - Connect to Qdrant and ensure collection exists.
    - Load cross-encoder model into memory.
    - Initialise embedder and sparse encoder.
    - Inject components into retriever node.

    Shutdown:
    - Close Qdrant connection pool.
    """
    settings = get_settings()
    log = logger.bind(collection=settings.qdrant_collection_name)

    log.info("Starting up RAG system")

    # Qdrant
    await ensure_collection_exists()
    log.info("Qdrant ready")

    # Retrieval components — load once, reuse across requests
    embedder = AsyncEmbedder()
    sparse_encoder = SparseEncoder()
    reranker = CrossEncoderReranker()

    # Inject into retriever node so it doesn't re-initialise per request
    init_retrieval_components(embedder, sparse_encoder, reranker)

    app.state.embedder = embedder
    app.state.sparse_encoder = sparse_encoder
    app.state.reranker = reranker

    log.info("All components initialised — server ready")

    yield

    # Shutdown
    log.info("Shutting down")
    await QdrantClientSingleton.close()
    log.info("Qdrant connection closed")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured :class:`FastAPI` instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Multimodal PDF RAG System",
        description="Agentic RAG over 1000+ PDFs with hybrid search and hallucination guards",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ------------------------------------------------------------------ #
    # Middleware (order matters — added in reverse execution order)
    # ------------------------------------------------------------------ #
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #
    app.include_router(query_router)
    app.include_router(ingest_router)

    # ------------------------------------------------------------------ #
    # Global exception handler
    # ------------------------------------------------------------------ #
    @app.exception_handler(RagException)
    async def rag_exception_handler(request: Request, exc: RagException) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            "RagException",
            error_code=exc.error_code,
            message=exc.message,
            request_id=request_id,
        )
        body = exc.to_dict()
        body["request_id"] = request_id
        return JSONResponse(status_code=exc.status_code, content=body)

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.error(
            "Unhandled exception",
            error=str(exc),
            request_id=request_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "internal_error",
                "message": "An unexpected error occurred",
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    # ------------------------------------------------------------------ #
    # Serve UI at root
    # ------------------------------------------------------------------ #
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def serve_ui() -> HTMLResponse:
        ui_path = __file__.replace("main.py", "templates/index.html")
        try:
            with open(ui_path, encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        except FileNotFoundError:
            return HTMLResponse(content="<h1>UI not found</h1>", status_code=404)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
