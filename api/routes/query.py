"""
api/routes/query.py

Query routes with session-based chat history.

  CHAT-1  In-memory session store: each browser session gets a unique
          session_id (from X-Session-ID header or auto-generated).
          Chat history is stored per session, max 20 exchanges.

  CHAT-2  History passed into AgentState so the decomposer can resolve
          references across turns.

  CHAT-3  History updated after each response.

  CHAT-4  History reset endpoint: DELETE /api/session.

Fixes:
  M-2   asyncio.Lock per session — eliminates race condition on concurrent
        requests sharing the same session_id.
  M-3   Background periodic cleanup task evicts expired sessions regardless
        of whether new requests arrive (prevents unbounded memory growth).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent.graph import rag_graph
from agent.state import AgentState, ChatMessage
from api.exceptions import RagException
from api.routes.metrics import record_query
from api.schemas import (
    CitationResponse,
    ImageResponse,
    QueryRequest,
    QueryResponse,
    TableResponse,
)

router = APIRouter(prefix="/api", tags=["query"])
logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ── Session store ─────────────────────────────────────────────────────────────
_sessions: dict[str, list[ChatMessage]] = defaultdict(list)
_session_timestamps: dict[str, float] = {}
_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
_MAX_HISTORY = 20
_SESSION_TTL = 7200  # 2 hours

# Background cleanup task handle
_cleanup_task: asyncio.Task | None = None


async def _periodic_cleanup() -> None:
    """Evict expired sessions every 10 minutes (M-3).

    Runs as a background asyncio task started at router import time.
    Ensures sessions are purged even if no new requests arrive.
    """
    while True:
        await asyncio.sleep(600)  # 10 minutes
        now = time.time()
        expired = [
            sid for sid, ts in list(_session_timestamps.items())
            if now - ts > _SESSION_TTL
        ]
        for sid in expired:
            _sessions.pop(sid, None)
            _session_timestamps.pop(sid, None)
            _session_locks.pop(sid, None)
        if expired:
            logger.debug("session.cleanup", evicted=len(expired))


def _ensure_cleanup_running() -> None:
    """Start the background cleanup task if not already running."""
    global _cleanup_task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = loop.create_task(_periodic_cleanup())


def _get_session_id(request: Request) -> str:
    return request.headers.get("X-Session-ID") or str(uuid.uuid4())


async def _get_history(session_id: str) -> list[ChatMessage]:
    """Get chat history for a session (M-2: lock-protected read)."""
    _ensure_cleanup_running()
    async with _session_locks[session_id]:
        now = time.time()
        _session_timestamps[session_id] = now
        return list(_sessions[session_id])


async def _update_history(session_id: str, query: str, answer: str) -> None:
    """Append Q&A pair to session history (M-2: lock-protected write)."""
    async with _session_locks[session_id]:
        history = _sessions[session_id]
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        if len(history) > _MAX_HISTORY:
            _sessions[session_id] = history[-_MAX_HISTORY:]
        _session_timestamps[session_id] = time.time()


def _build_query_response(
    result: dict,
    request_id: str,
    query: str,
) -> QueryResponse:
    """Build QueryResponse from agent result dict."""
    answer = result.get("final_answer") or result.get("generated_answer", "")
    citations = [CitationResponse(**c) for c in result.get("citations", [])]

    images = [
        ImageResponse(
            filename=img["filename"],
            page=img["page"],
            caption=img.get("caption", ""),
            image_b64=img.get("image_b64"),
        )
        for img in result.get("retrieved_images", [])
    ]

    tables = [
        TableResponse(
            markdown=tbl["markdown"],
            caption=tbl.get("caption", ""),
        )
        for tbl in result.get("retrieved_tables", [])
    ]

    return QueryResponse(
        answer=answer,
        citations=citations,
        faithfulness_score=float(result.get("faithfulness_score", 0.0)),
        consistency_passed=bool(result.get("consistency_passed", False)),
        sub_queries=result.get("sub_queries", [query]),
        request_id=request_id,
        images=images,
        tables=tables,
        completeness_score=float(result.get("completeness_score", 1.0)),
        query_type=result.get("query_type", "analytical"),
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Run the full agentic RAG pipeline with chat history."""
    request_id = getattr(request.state, "request_id", "unknown")
    session_id = _get_session_id(request)
    chat_history = await _get_history(session_id)

    log = logger.bind(
        request_id=request_id,
        session_id=session_id[:8],
        query=body.query[:80],
        history_turns=len(chat_history) // 2,
    )
    log.info("Query received")

    t0 = time.time()
    success = False
    result: dict = {}

    initial_state: AgentState = {
        "original_query": body.query,
        "chat_history": chat_history,
        "retry_count": 0,
    }

    try:
        result = await rag_graph.ainvoke(initial_state)
        success = True
    except RagException:
        raise
    except Exception as exc:
        log.error("Unexpected agent error", error=str(exc))
        raise
    finally:
        latency = time.time() - t0
        faithfulness = float(result.get("faithfulness_score", 0.0)) if success else 0.0
        chunks = len(result.get("reranked_chunks") or []) if success else 0
        record_query(
            success=success,
            latency_seconds=latency,
            faithfulness_score=faithfulness,
            chunks_retrieved=chunks,
        )

    response = _build_query_response(result, request_id, body.query)

    if success:
        await _update_history(session_id, body.query, response.answer)

    log.info(
        "Query complete",
        answer_length=len(response.answer),
        citations=len(response.citations),
        faithfulness=response.faithfulness_score,
        latency=round(time.time() - t0, 2),
    )

    return response


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """Stream the RAG pipeline response as Server-Sent Events with history."""
    request_id = getattr(request.state, "request_id", "unknown")
    session_id = _get_session_id(request)
    chat_history = await _get_history(session_id)

    log = logger.bind(
        request_id=request_id,
        session_id=session_id[:8],
        query=body.query[:80],
        history_turns=len(chat_history) // 2,
    )
    log.info("Streaming query received")

    async def event_stream():
        t0 = time.time()
        success = False
        result: dict = {}
        full_answer = ""

        def sse(event: str, data: object) -> str:
            payload = json.dumps(data) if not isinstance(data, str) else data
            return f"event: {event}\ndata: {payload}\n\n"

        try:
            # Signal pipeline start
            yield sse("status", {"stage": "decomposing", "message": "Decomposing your query..."})
            await asyncio.sleep(0)

            initial_state: AgentState = {
                "original_query": body.query,
                "chat_history": chat_history,
                "retry_count": 0,
            }

            # Run the full pipeline
            result = await rag_graph.ainvoke(initial_state)
            success = True

            # Post-pipeline status events — reflect what already happened
            yield sse("status", {"stage": "retrieved", "message": "Documents retrieved and reranked."})
            await asyncio.sleep(0)
            yield sse("status", {"stage": "verified", "message": "Answer verified for faithfulness."})
            await asyncio.sleep(0)

            # Stream answer word by word
            answer = result.get("final_answer") or result.get("generated_answer", "")
            full_answer = answer
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield sse("chunk", {"text": token})
                await asyncio.sleep(0.02)

            yield sse("citations", result.get("citations", []))

            retrieved_images = result.get("retrieved_images", [])
            if retrieved_images:
                yield sse("images", retrieved_images)

            retrieved_tables = result.get("retrieved_tables", [])
            if retrieved_tables:
                yield sse("tables", retrieved_tables)

            yield sse("scores", {
                "faithfulness_score": float(result.get("faithfulness_score", 0.0)),
                "consistency_passed": bool(result.get("consistency_passed", False)),
                "completeness_score": float(result.get("completeness_score", 1.0)),
                "sub_queries": result.get("sub_queries", [body.query]),
                "rewritten_query": result.get("rewritten_query", body.query),
                "query_type": result.get("query_type", "analytical"),
                "request_id": request_id,
            })

            yield sse("done", {"message": "Stream complete"})
            log.info("Stream complete", latency=round(time.time() - t0, 2))

        except Exception as exc:
            log.error("Streaming query failed", error=str(exc))
            yield sse("error", {"message": "An error occurred during query processing."})

        finally:
            latency = time.time() - t0
            faithfulness = float(result.get("faithfulness_score", 0.0)) if success else 0.0
            chunks = len(result.get("reranked_chunks") or []) if success else 0
            record_query(
                success=success,
                latency_seconds=latency,
                faithfulness_score=faithfulness,
                chunks_retrieved=chunks,
            )
            if success and full_answer:
                await _update_history(session_id, body.query, full_answer)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
            "X-Session-ID": session_id,
        },
    )


@router.delete("/session")
async def clear_session(request: Request) -> JSONResponse:
    """Clear chat history for the current session."""
    session_id = _get_session_id(request)
    async with _session_locks[session_id]:
        _sessions.pop(session_id, None)
        _session_timestamps.pop(session_id, None)
    _session_locks.pop(session_id, None)
    logger.info("session.cleared", session_id=session_id[:8])
    return JSONResponse({"status": "cleared", "session_id": session_id})
