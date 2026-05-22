"""
agent/nodes/retriever.py

Advanced retrieval node integrating:
  B1  — HyDE: embed a hypothetical answer for dense retrieval
  B3  — MMR: diversify final chunk selection after reranking
  NEW — RAG Fusion: generate N paraphrases → retrieve → client-side RRF merge
  NEW — Step-Back: generate abstract query for broader context retrieval
  NEW — Sentence Window: expand retrieved chunks to neighboring context
  NEW — Contextual Compression: extract query-relevant sentences (optional)
  NEW — Query-Type routing: adjust retrieval strategy per query type

Original fixes preserved:
  FIX-1  Complex query detection → dynamic top_k scaling
  FIX-2  Section sub-query injection for explicit structural references
  FIX-3  TOP_K_FINAL scaled for complex queries
  FIX-4  Text-hash deduplication before returning to generator
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any

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
_openai_client: Any | None = None


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


def set_retrieval_openai_client(client: Any) -> None:
    """Inject the shared AsyncOpenAI client (called from main.py lifespan)."""
    global _openai_client
    _openai_client = client


def _get_components() -> tuple[AsyncEmbedder, SparseEncoder, CrossEncoderReranker]:
    global _embedder, _sparse_encoder, _reranker
    if _embedder is None:
        _embedder = AsyncEmbedder()
    if _sparse_encoder is None:
        _sparse_encoder = SparseEncoder()
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _embedder, _sparse_encoder, _reranker


def _get_openai_client() -> Any | None:
    """Return the OpenAI client if set, else None (features degrade gracefully)."""
    return _openai_client


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


def _apply_query_type_filters(
    query_type: str,
    retriever: HybridRetriever,
) -> dict:
    """Return keyword args to pass to retriever.retrieve_multi based on query type."""
    from retrieval.filters import build_filter
    extra: dict = {}
    if query_type == "visual":
        extra["filters"] = build_filter(content_types=["image", "chart"])
    elif query_type == "table":
        extra["filters"] = build_filter(content_types=["table"])
    return extra


async def retriever_node(state: AgentState) -> AgentState:
    """Advanced retrieval node with RAG Fusion, Step-Back, Sentence Window, and Compression.

    Pipeline:
    1.  Detect complex query → scale top_k
    2.  Classify query type → routing decisions
    3.  In parallel: HyDE + Step-Back + RAG Fusion variant generation
    4.  Build unified query pool (deduped)
    5.  Parallel hybrid retrieval across entire query pool
    6.  Cross-encoder reranking
    7.  MMR diversification
    8.  Sentence Window expansion (context enrichment)
    9.  Contextual Compression (optional, configurable)
    10. Text-hash deduplication
    """
    settings = get_settings()
    original_query = state["original_query"]
    sub_queries: list[str] = state.get("sub_queries") or [original_query]
    log = logger.bind(node="retriever", query=original_query[:80])

    is_complex = _is_complex_query(original_query)
    query_type = state.get("query_type") or "analytical"

    # For visual/table queries, limit top_k (precision > recall)
    if query_type in ("visual", "table"):
        retrieval_top_k = settings.max_retrieval_chunks
        final_top_k = settings.top_k_final
    else:
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
        query_type=query_type,
        is_complex=is_complex,
        retrieval_top_k=retrieval_top_k,
        final_top_k=final_top_k,
    )

    # ── Expand with section sub-queries ──────────────────────────────────────
    expanded = list(sub_queries)
    for q in sub_queries:
        expanded.extend(_extract_section_sub_queries(q, original_query))
    seen_q: set[str] = set()
    base_queries: list[str] = []
    for q in expanded:
        if q not in seen_q:
            seen_q.add(q)
            base_queries.append(q)

    embedder, sparse_encoder, reranker = _get_components()
    openai_client = _get_openai_client()

    # ── Parallel generation: HyDE + Step-Back + RAG Fusion ───────────────────
    hyde_doc = ""
    step_back_q = ""
    fusion_variants: list[str] = []

    async def _gen_hyde() -> str:
        if not settings.hyde_enabled or not openai_client:
            return ""
        try:
            from retrieval.hyde import generate_hypothetical_document
            doc = await generate_hypothetical_document(original_query)
            log.debug("retriever_node.hyde_done", length=len(doc))
            return doc
        except Exception as exc:
            log.warning("retriever_node.hyde_failed", error=str(exc))
            return ""

    async def _gen_step_back() -> str:
        if not settings.step_back_enabled or not openai_client:
            return ""
        try:
            from retrieval.step_back import generate_step_back_query
            q = await generate_step_back_query(original_query, openai_client)
            if q:
                log.info("retriever_node.step_back_done", step_back=q[:80])
            return q
        except Exception as exc:
            log.warning("retriever_node.step_back_failed", error=str(exc))
            return ""

    async def _gen_fusion() -> list[str]:
        if not settings.rag_fusion_enabled or not openai_client:
            return []
        try:
            from retrieval.rag_fusion import generate_rag_fusion_queries
            variants = await generate_rag_fusion_queries(
                original_query,
                n=settings.rag_fusion_num_queries,
                openai_client=openai_client,
            )
            log.info("retriever_node.fusion_done", variants=len(variants))
            return variants
        except Exception as exc:
            log.warning("retriever_node.fusion_failed", error=str(exc))
            return []

    hyde_doc, step_back_q, fusion_variants = await asyncio.gather(
        _gen_hyde(), _gen_step_back(), _gen_fusion()
    )

    # ── Assemble unified query pool ───────────────────────────────────────────
    # Order matters: HyDE first (best embedding proxy), then base queries,
    # then step-back (broad context), then fusion variants (recall diversity)
    all_queries: list[str] = []
    if hyde_doc:
        all_queries.append(hyde_doc)
    all_queries.extend(base_queries)
    if step_back_q and step_back_q not in seen_q:
        all_queries.append(step_back_q)
    for fq in fusion_variants:
        if fq not in seen_q and fq not in all_queries:
            all_queries.append(fq)

    # Scale top_k_per_query down when many queries to keep total candidates reasonable
    target_candidates = 120 if is_complex else 80
    top_k_per_query = max(10, target_candidates // max(len(all_queries), 1))

    log.info(
        "retriever_node.query_pool",
        total_queries=len(all_queries),
        top_k_per_query=top_k_per_query,
        has_hyde=bool(hyde_doc),
        has_step_back=bool(step_back_q),
        fusion_variants=len(fusion_variants),
    )

    from retrieval.qdrant_client import QdrantClientSingleton
    client = await QdrantClientSingleton.get()
    retriever = HybridRetriever(client, settings, embedder, sparse_encoder)

    # Apply content-type filters for visual/table queries
    extra_kwargs = _apply_query_type_filters(query_type, retriever)

    raw_dicts = await retriever.retrieve_multi(
        queries=all_queries,
        top_k_per_query=top_k_per_query,
        **extra_kwargs,
    )

    if not raw_dicts:
        raise RetrievalError(
            "All sub-query retrievals returned empty results",
            detail=f"sub_queries={base_queries}",
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

    # ── Cross-encoder reranking ───────────────────────────────────────────────
    reranked = await reranker.rerank(
        query=original_query,
        chunks=retrieved,
        top_k=final_top_k,
    )

    # ── MMR diversification — reuses stored Qdrant vectors (no API call) ────
    if settings.mmr_enabled and len(reranked) > 1:
        try:
            from retrieval.mmr import mmr_select

            # Fetch stored dense vectors by ID — one local Qdrant round-trip
            # instead of an OpenAI embedding API call.
            chunk_ids = [c.id for c in reranked]
            records = await client.retrieve(
                collection_name=settings.qdrant_collection_name,
                ids=chunk_ids,
                with_vectors=["dense"],
            )
            vec_by_id: dict[str, list[float]] = {
                str(r.id): r.vector["dense"]  # type: ignore[index]
                for r in records
                if r.vector and isinstance(r.vector, dict) and "dense" in r.vector
            }
            chunk_embeddings: list[list[float]] = [
                vec_by_id.get(str(c.id), []) for c in reranked
            ]

            # Fallback: re-embed any chunk whose vector wasn't returned
            missing_idx = [i for i, e in enumerate(chunk_embeddings) if not e]
            if missing_idx:
                fallback_vecs = await embedder.embed(
                    [reranked[i].text[:500] for i in missing_idx]
                )
                for i, vec in zip(missing_idx, fallback_vecs):
                    chunk_embeddings[i] = vec

            reranked = mmr_select(
                chunks=reranked,
                embeddings=chunk_embeddings,
                top_k=final_top_k,
                lambda_param=settings.mmr_lambda,
            )
            log.info("retriever_node.mmr_done", count=len(reranked))
        except Exception as exc:
            log.warning("retriever_node.mmr_failed", error=str(exc))

    # ── Sentence Window expansion ─────────────────────────────────────────────
    # Only expand text chunks — tables/images don't benefit and may have no neighbors
    if settings.sentence_window_enabled:
        try:
            text_chunks = [c for c in reranked if c.content_type == "text"]
            non_text = [c for c in reranked if c.content_type != "text"]

            if text_chunks:
                from retrieval.sentence_window import expand_chunks_to_window
                expanded_text = await expand_chunks_to_window(
                    chunks=text_chunks,
                    client=client,
                    collection=settings.qdrant_collection_name,
                    window_size=settings.sentence_window_size,
                )
                # Re-merge preserving original order
                text_by_id = {c.id: c for c in expanded_text}
                reranked = [
                    text_by_id.get(c.id, c) if c.content_type == "text" else c
                    for c in reranked
                ]
                log.info("retriever_node.window_done")
        except Exception as exc:
            log.warning("retriever_node.window_failed", error=str(exc))

    # ── Contextual Compression (optional) ────────────────────────────────────
    if settings.contextual_compression_enabled and openai_client:
        try:
            from retrieval.contextual_compressor import compress_chunks_batch
            reranked = await compress_chunks_batch(
                query=original_query,
                chunks=reranked,
                openai_client=openai_client,
                model=settings.contextual_compression_model,
            )
            log.info("retriever_node.compression_done")
        except Exception as exc:
            log.warning("retriever_node.compression_failed", error=str(exc))

    # ── Dedup by text ─────────────────────────────────────────────────────────
    reranked = _dedup_by_text(reranked)
    log.info("retriever_node.final", count=len(reranked))

    return {
        **state,
        "retrieved_chunks": retrieved,
        "reranked_chunks": reranked,
        "step_back_query": step_back_q,
        "fusion_queries": fusion_variants,
    }
