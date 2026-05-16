from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from agent.graph import rag_graph
from agent.state import AgentState
from api.exceptions import RagException
from api.schemas import CitationResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/api", tags=["query"])
logger: structlog.BoundLogger = structlog.get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Run the full agentic RAG pipeline for a user query.

    Invokes the LangGraph agent: decompose → retrieve → rerank →
    generate → verify. Returns a grounded, cited answer.

    Args:
        request: FastAPI request (used for request_id).
        body: Validated query payload.

    Returns:
        :class:`QueryResponse` with answer, citations, and quality scores.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    log = logger.bind(request_id=request_id, query=body.query[:80])
    log.info("Query received")

    initial_state: AgentState = {
        "original_query": body.query,
        "retry_count": 0,
    }

    try:
        result = await rag_graph.ainvoke(initial_state)
    except RagException:
        raise
    except Exception as exc:
        log.error("Unexpected agent error", error=str(exc))
        raise

    answer = result.get("final_answer") or result.get("generated_answer", "")
    citations = [
        CitationResponse(**c) for c in result.get("citations", [])
    ]

    log.info(
        "Query complete",
        answer_length=len(answer),
        citations=len(citations),
        faithfulness=result.get("faithfulness_score", 0.0),
    )

    return QueryResponse(
        answer=answer,
        citations=citations,
        faithfulness_score=result.get("faithfulness_score", 0.0),
        consistency_passed=result.get("consistency_passed", False),
        sub_queries=result.get("sub_queries", [body.query]),
        request_id=request_id,
    )
