from __future__ import annotations

import asyncio

import structlog

from agent.state import AgentState
from api.exceptions import RetrievalError
from ingestion.embedder import AsyncEmbedder
from ingestion.sparse_encoder import SparseEncoder
from retrieval.filters import build_filter
from retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from retrieval.reranker import CrossEncoderReranker

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Module-level singletons — initialised once, reused across requests.
# In FastAPI, these are set on app.state at lifespan startup instead.
_embedder: AsyncEmbedder | None = None
_sparse_encoder: SparseEncoder | None = None
_reranker: CrossEncoderReranker | None = None


def init_retrieval_components(
    embedder: AsyncEmbedder,
    sparse_encoder: SparseEncoder,
    reranker: CrossEncoderReranker,
) -> None:
    """Inject pre-built retrieval components (called from FastAPI lifespan).

    Args:
        embedder: Initialised :class:`AsyncEmbedder`.
        sparse_encoder: Initialised :class:`SparseEncoder`.
        reranker: Initialised :class:`CrossEncoderReranker`.
    """
    global _embedder, _sparse_encoder, _reranker
    _embedder = embedder
    _sparse_encoder = sparse_encoder
    _reranker = reranker


def _get_components() -> tuple[AsyncEmbedder, SparseEncoder, CrossEncoderReranker]:
    """Return retrieval components, creating them lazily if not injected.

    Returns:
        Tuple of (embedder, sparse_encoder, reranker).
    """
    global _embedder, _sparse_encoder, _reranker
    if _embedder is None:
        _embedder = AsyncEmbedder()
    if _sparse_encoder is None:
        _sparse_encoder = SparseEncoder()
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _embedder, _sparse_encoder, _reranker


def _dedup_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicate chunks by chunk_id, keeping highest score.

    Args:
        chunks: Mixed list of chunks from multiple sub-query retrievals.

    Returns:
        Deduplicated list sorted by score descending.
    """
    seen: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        if chunk.chunk_id not in seen or chunk.score > seen[chunk.chunk_id].score:
            seen[chunk.chunk_id] = chunk
    return sorted(seen.values(), key=lambda c: c.score, reverse=True)


async def retriever_node(state: AgentState) -> AgentState:
    """Retrieve and re-rank chunks for all sub-queries in parallel.

    Runs hybrid retrieval for each sub-query concurrently using
    ``asyncio.gather``, deduplicates results by chunk_id, then re-ranks
    the combined candidate pool with the cross-encoder.

    Args:
        state: Current agent state. Reads ``sub_queries``.

    Returns:
        Updated state with ``retrieved_chunks`` and ``reranked_chunks``.

    Raises:
        RetrievalError: If retrieval fails for all sub-queries.
    """
    sub_queries = state.get("sub_queries") or [state["original_query"]]
    log = logger.bind(node="retriever", sub_queries=len(sub_queries))
    log.info("Starting parallel retrieval")

    embedder, sparse_encoder, reranker = _get_components()
    retriever = HybridRetriever(embedder, sparse_encoder)

    # Run all sub-query retrievals concurrently.
    async def _retrieve_one(q: str) -> list[RetrievedChunk]:
        try:
            return await retriever.retrieve(q, query_filter=build_filter())
        except Exception as exc:
            log.warning("Sub-query retrieval failed", query=q[:60], error=str(exc))
            return []

    results = await asyncio.gather(*[_retrieve_one(q) for q in sub_queries])

    # Flatten and dedup across all sub-query results.
    all_chunks: list[RetrievedChunk] = []
    for chunk_list in results:
        all_chunks.extend(chunk_list)

    if not all_chunks:
        raise RetrievalError(
            "All sub-query retrievals returned empty results",
            detail=f"sub_queries={sub_queries}",
        )

    deduped = _dedup_chunks(all_chunks)
    log.info("Retrieval complete", total=len(all_chunks), after_dedup=len(deduped))

    # Re-rank with cross-encoder using the original query for scoring.
    reranked = await reranker.rerank(
        state["original_query"],
        deduped,
    )

    return {
        **state,
        "retrieved_chunks": deduped,
        "reranked_chunks": reranked,
    }
