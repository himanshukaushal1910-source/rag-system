"""
agent/nodes/retriever.py

Retrieval node — now integrates HyDE (B1) and MMR (B3).

CANONICAL version — replaces both retriever.py and retriever_node.py.

New features:
  B1 — HyDE: embed a hypothetical answer instead of raw query for dense search
  B3 — MMR: diversify final chunk selection after reranking

All original fixes preserved:
  FIX-1  Complex query detection → dynamic top_k scaling
  FIX-2  Section sub-query injection for explicit references
  FIX-3  TOP_K_FINAL scaled for complex queries
  FIX-4  Text-hash deduplication before returning to generator
  FIX-8  HybridRetriever constructor (client, settings, embedder, sparse_encoder)
"""

from __future__ import annotations

import asyncio
import hashlib
import re

import structlog

from agent.state import AgentState
from api.exceptions import RetrievalError
from config import get_settings
from ingestion.embedder import AsyncEmbedder
from ingestion.sparse_encoder import SparseEncoder
from retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from retrieval.reranker import CrossEncoderReranker

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_COMPLEX_QUERY_RE = re.compile(
    r"\b(explain|analyze|analyse|compare|contrast|detail|describe|"
    r"comprehensive|complete|all|every|full|entire|discuss|evaluate|"
    r"provide|summarize|summarise|outline|overview)\b",
    re.I,
)
_SECTION_REF_RE = re.compile(
    r"\b(section|table|figure|fig|algorithm|equation|appendix)\s*([\d\.]+)\b",
    re.I,
)

_embedder: AsyncEmbedder | None = None
_sparse_encoder: SparseEncoder | None = None
_reranker: CrossEncoderReranker | None = None


def init_retrieval_components(
    embedder: AsyncEmbedder,
    sparse_encoder: SparseEncoder,
    reranker: CrossEncoderReranker,
) -> None:
    """Inject pre-built components from FastAPI lifespan."""
    global _embedder, _sparse_encoder, _reranker
    _embedder = embedder
    _sparse_encoder = sparse_encoder
    _reranker = reranker


def _get_components() -> tuple[AsyncEmbedder, SparseEncoder, CrossEncoderReranker]:
    global _embedder, _sparse_encoder, _reranker
    if _embedder is None:
        _embedder = AsyncEmbedder()
    if _sparse_encoder is None:
        _sparse_encoder = SparseEncoder()
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _embedder, _sparse_encoder, _reranker


def _is_complex_query(query: str) -> bool:
    return bool(_COMPLEX_QUERY_RE.search(query))


def _extract_section_sub_queries(query: str, original_query: str) -> list[str]:
    sub_queries: list[str] = []
    for match in _SECTION_REF_RE.finditer(query):
        ref_type = match.group(1)
        ref_num = match.group(2)
        sub_queries.append(f"{ref_type} {ref_num} {original_query[:60]}")
    return sub_queries


def _dedup_by_text(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    result: list[RetrievedChunk] = []
    for chunk in chunks:
        text_hash = hashlib.md5(chunk.text[:200].encode()).hexdigest()
        if text_hash not in seen:
            seen.add(text_hash)
            result.append(chunk)
    return result


async def retriever_node(state: AgentState) -> AgentState:
    """Retrieve and rerank chunks for all sub-queries.

    Pipeline:
    1. HyDE — generate hypothetical doc for better dense embedding (B1)
    2. Parallel retrieval — all sub-queries retrieved simultaneously
    3. Cross-encoder reranking
    4. MMR diversification (B3)
    5. Text-hash deduplication

    Args:
        state: Current agent state with sub_queries and original_query.

    Returns:
        Updated state with retrieved_chunks and reranked_chunks.
    """
    settings = get_settings()
    original_query = state["original_query"]
    sub_queries: list[str] = state.get("sub_queries") or [original_query]
    log = logger.bind(node="retriever", query=original_query[:80])

    is_complex = _is_complex_query(original_query)
    retrieval_top_k = (
        settings.max_retrieval_chunks * 2 if is_complex
        else settings.max_retrieval_chunks
    )
    final_top_k = (
        settings.top_k_final * 2 if is_complex
        else settings.top_k_final
    )

    log.info(
        "retriever_node.start",
        sub_queries=len(sub_queries),
        is_complex=is_complex,
        retrieval_top_k=retrieval_top_k,
        final_top_k=final_top_k,
    )

    # ── B2: expand with section sub-queries ──────────────────────────────
    expanded = list(sub_queries)
    for q in sub_queries:
        expanded.extend(_extract_section_sub_queries(q, original_query))
    seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in expanded:
        if q not in seen_q:
            seen_q.add(q)
            unique_queries.append(q)

    embedder, sparse_encoder, reranker = _get_components()

    # ── B1: HyDE — generate hypothetical doc and prepend as extra query ────
    # HyDE augments retrieval without discarding any decomposed sub-query.
    # The hypothetical document is added as an ADDITIONAL query so all
    # original sub-queries remain in the retrieval set (M-6 fix).
    hyde_queries = list(unique_queries)
    if settings.hyde_enabled:
        try:
            from retrieval.hyde import generate_hypothetical_document
            hyde_doc = await generate_hypothetical_document(original_query)
            log.debug("retriever_node.hyde_done", length=len(hyde_doc))
            # Prepend — first query gets most weight in fusion scoring
            hyde_queries = [hyde_doc] + unique_queries
        except Exception as exc:
            log.warning("retriever_node.hyde_failed", error=str(exc))

    from retrieval.qdrant_client import QdrantClientSingleton
    client = await QdrantClientSingleton.get()
    retriever = HybridRetriever(client, settings, embedder, sparse_encoder)

    raw_dicts = await retriever.retrieve_multi(
        queries=hyde_queries,
        top_k_per_query=retrieval_top_k,
    )

    if not raw_dicts:
        raise RetrievalError(
            "All sub-query retrievals returned empty results",
            detail=f"sub_queries={unique_queries}",
        )

    retrieved: list[RetrievedChunk] = [
        RetrievedChunk(
            id=item["id"],
            score=item["score"],
            text=item["payload"].get("text", ""),
            filename=item["payload"].get("filename", ""),
            page_number=item["payload"].get("page_number", 0),
            content_type=item["payload"].get("content_type", "text"),
            image_b64=item["payload"].get("image_b64"),
            section_heading=item["payload"].get("section_heading", ""),
        )
        for item in raw_dicts
    ]

    log.info("retriever_node.retrieved", count=len(retrieved))

    # Rerank
    reranked = await reranker.rerank(
        query=original_query,
        chunks=retrieved,
        top_k=final_top_k,
    )

    # ── B3: MMR diversification ───────────────────────────────────────────
    if settings.mmr_enabled and len(reranked) > 1:
        try:
            # Embed reranked chunk texts for MMR similarity computation
            chunk_texts = [c.text[:500] for c in reranked]
            chunk_embeddings = await embedder.embed(chunk_texts)

            from retrieval.mmr import mmr_select
            reranked = mmr_select(
                chunks=reranked,
                embeddings=chunk_embeddings,
                top_k=final_top_k,
                lambda_param=settings.mmr_lambda,
            )
            log.info("retriever_node.mmr_done", count=len(reranked))
        except Exception as exc:
            log.warning("retriever_node.mmr_failed", error=str(exc))

    # Dedup by text
    reranked = _dedup_by_text(reranked)
    log.info("retriever_node.final", count=len(reranked))

    return {
        **state,
        "retrieved_chunks": retrieved,
        "reranked_chunks": reranked,
    }