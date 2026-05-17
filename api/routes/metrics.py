from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from config import get_settings
from retrieval.qdrant_client import QdrantClientSingleton

router = APIRouter(tags=["observability"])
logger: structlog.BoundLogger = structlog.get_logger(__name__)

# In-memory metrics store
_metrics: dict[str, Any] = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "requests_rate_limited": 0,
    "query_latency_sum": 0.0,
    "query_latency_count": 0,
    "faithfulness_sum": 0.0,
    "faithfulness_count": 0,
    "chunks_retrieved_sum": 0,
    "chunks_retrieved_count": 0,
    "start_time": time.time(),
}

# Per-endpoint counters
_endpoint_counts: dict[str, int] = defaultdict(int)


def record_query(
    *,
    success: bool,
    latency_seconds: float,
    faithfulness_score: float,
    chunks_retrieved: int,
) -> None:
    """Record metrics for a completed query.

    Called from the query route after each request completes.

    Args:
        success: Whether the query completed without error.
        latency_seconds: End-to-end latency in seconds.
        faithfulness_score: Faithfulness score from verifier (0-1).
        chunks_retrieved: Number of chunks after reranking.
    """
    _metrics["requests_total"] += 1
    if success:
        _metrics["requests_success"] += 1
    else:
        _metrics["requests_failed"] += 1

    _metrics["query_latency_sum"] += latency_seconds
    _metrics["query_latency_count"] += 1

    if faithfulness_score > 0:
        _metrics["faithfulness_sum"] += faithfulness_score
        _metrics["faithfulness_count"] += 1

    _metrics["chunks_retrieved_sum"] += chunks_retrieved
    _metrics["chunks_retrieved_count"] += 1


def record_rate_limited() -> None:
    """Increment rate-limited request counter."""
    _metrics["requests_rate_limited"] += 1


def _prometheus_gauge(name: str, value: float, help_text: str, labels: str = "") -> str:
    label_str = f"{{{labels}}}" if labels else ""
    return (
        f"# HELP {name} {help_text}\n"
        f"# TYPE {name} gauge\n"
        f"{name}{label_str} {value}\n"
    )


def _prometheus_counter(name: str, value: float, help_text: str, labels: str = "") -> str:
    label_str = f"{{{labels}}}" if labels else ""
    return (
        f"# HELP {name} {help_text}\n"
        f"# TYPE {name} counter\n"
        f"{name}{label_str} {value}\n"
    )


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics() -> str:
    """Prometheus-compatible metrics endpoint.

    Returns metrics in Prometheus text exposition format.
    Scrape with Prometheus or view raw in browser.
    """
    settings = get_settings()
    uptime = time.time() - _metrics["start_time"]

    # Qdrant chunk count
    chunk_count = 0
    try:
        client = await QdrantClientSingleton.get()
        info = await client.get_collection(settings.qdrant_collection_name)
        chunk_count = info.points_count or 0
    except Exception:
        pass

    # Derived metrics
    avg_latency = (
        _metrics["query_latency_sum"] / _metrics["query_latency_count"]
        if _metrics["query_latency_count"] > 0 else 0.0
    )
    avg_faithfulness = (
        _metrics["faithfulness_sum"] / _metrics["faithfulness_count"]
        if _metrics["faithfulness_count"] > 0 else 0.0
    )
    avg_chunks = (
        _metrics["chunks_retrieved_sum"] / _metrics["chunks_retrieved_count"]
        if _metrics["chunks_retrieved_count"] > 0 else 0.0
    )
    success_rate = (
        _metrics["requests_success"] / _metrics["requests_total"]
        if _metrics["requests_total"] > 0 else 1.0
    )

    lines = [
        "# RAG System Metrics\n",
        _prometheus_counter("rag_requests_total", _metrics["requests_total"], "Total query requests"),
        _prometheus_counter("rag_requests_success_total", _metrics["requests_success"], "Successful query requests"),
        _prometheus_counter("rag_requests_failed_total", _metrics["requests_failed"], "Failed query requests"),
        _prometheus_counter("rag_requests_rate_limited_total", _metrics["requests_rate_limited"], "Rate limited requests"),
        _prometheus_gauge("rag_request_success_rate", round(success_rate, 4), "Request success rate (0-1)"),
        _prometheus_gauge("rag_query_latency_avg_seconds", round(avg_latency, 3), "Average query latency in seconds"),
        _prometheus_gauge("rag_faithfulness_avg", round(avg_faithfulness, 4), "Average faithfulness score (0-1)"),
        _prometheus_gauge("rag_chunks_retrieved_avg", round(avg_chunks, 2), "Average chunks retrieved per query"),
        _prometheus_gauge("rag_chunks_indexed_total", chunk_count, "Total chunks indexed in Qdrant"),
        _prometheus_gauge("rag_uptime_seconds", round(uptime, 1), "Server uptime in seconds"),
    ]

    return "".join(lines)
