from __future__ import annotations

import asyncio
import json
import time

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from agent.graph import rag_graph
from agent.state import AgentState
from api.exceptions import RagException
from api.routes.metrics import record_query
from api.schemas import CitationResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/api", tags=["query"])
logger: structlog.BoundLogger = structlog.get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Run the full agentic RAG pipeline for a user query.

    Args:
        request: FastAPI request (used for request_id).
        body: Validated query payload.

    Returns:
        :class:`QueryResponse` with answer, citations, and quality scores.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    log = logger.bind(request_id=request_id, query=body.query[:80])
    log.info("Query received")

    t0 = time.time()
    success = False
    result: dict = {}

    initial_state: AgentState = {
        "original_query": body.query,
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

    answer = result.get("final_answer") or result.get("generated_answer", "")
    citations = [CitationResponse(**c) for c in result.get("citations", [])]

    log.info(
        "Query complete",
        answer_length=len(answer),
        citations=len(citations),
        faithfulness=result.get("faithfulness_score", 0.0),
        latency=round(time.time() - t0, 2),
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        faithfulness_score=float(result.get("faithfulness_score", 0.0)),
        consistency_passed=bool(result.get("consistency_passed", False)),
        sub_queries=result.get("sub_queries", [body.query]),
        request_id=request_id,
    )


@router.post("/query/stream")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    """Stream the RAG pipeline response as Server-Sent Events.

    Pipeline stages broadcast as status events, then the answer is
    streamed word-by-word as chunk events, followed by citations and scores.

    SSE event types in order:
    - ``status``: pipeline stage updates (decomposing, retrieving, generating, verifying)
    - ``chunk``: answer tokens streamed word by word
    - ``citations``: structured citations JSON array
    - ``scores``: faithfulness score, consistency, sub-queries
    - ``done``: stream complete signal
    - ``error``: on failure

    Args:
        request: FastAPI request.
        body: Validated query payload.

    Returns:
        Server-Sent Events stream.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    log = logger.bind(request_id=request_id, query=body.query[:80])
    log.info("Streaming query received")

    async def event_stream():
        t0 = time.time()
        success = False
        result: dict = {}

        def sse(event: str, data: object) -> str:
            payload = json.dumps(data) if not isinstance(data, str) else data
            return f"event: {event}\ndata: {payload}\n\n"

        try:
            yield sse("status", {"stage": "decomposing", "message": "Decomposing your query..."})
            await asyncio.sleep(0)

            initial_state: AgentState = {
                "original_query": body.query,
                "retry_count": 0,
            }

            yield sse("status", {"stage": "retrieving", "message": "Searching through documents..."})
            await asyncio.sleep(0)

            yield sse("status", {"stage": "generating", "message": "Generating grounded answer..."})
            await asyncio.sleep(0)

            # Run full pipeline
            result = await rag_graph.ainvoke(initial_state)
            success = True

            yield sse("status", {"stage": "verifying", "message": "Verifying answer faithfulness..."})
            await asyncio.sleep(0)

            # Stream answer word by word for typewriter effect
            answer = result.get("final_answer") or result.get("generated_answer", "")
            words = answer.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield sse("chunk", {"text": token})
                await asyncio.sleep(0.025)

            # Citations
            yield sse("citations", result.get("citations", []))

            # Scores
            yield sse("scores", {
                "faithfulness_score": float(result.get("faithfulness_score", 0.0)),
                "consistency_passed": bool(result.get("consistency_passed", False)),
                "sub_queries": result.get("sub_queries", [body.query]),
                "request_id": request_id,
            })

            yield sse("done", {"message": "Stream complete"})
            log.info("Stream complete", latency=round(time.time() - t0, 2))

        except Exception as exc:
            log.error("Streaming query failed", error=str(exc))
            yield sse("error", {"message": str(exc)})

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

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )
