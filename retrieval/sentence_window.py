"""
retrieval/sentence_window.py

Sentence-Window Context Expansion.

Instead of returning raw retrieved chunks, expand each chunk to include its
surrounding chunks from the same document and page neighborhood.

Strategy:
- For each retrieved chunk, query Qdrant for chunks from the same doc_id
  with chunk_index in [chunk_index - window, chunk_index + window]
- Merge their text (in order) as the expanded context
- Keep the original chunk's metadata (filename, page, score) for citations

Why this helps:
- Semantic similarity often fires on a specific sentence, but the answer
  requires the surrounding sentences for full context
- Especially valuable for numerical data (a number in sentence N, its explanation in N+1)
- No re-ingestion needed — works with existing Qdrant data

Note: We store chunk_index as a payload field during ingestion.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import structlog
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from retrieval.hybrid_retriever import RetrievedChunk

logger = structlog.get_logger(__name__)


async def _expand_single_chunk(
    chunk: RetrievedChunk,
    client: AsyncQdrantClient,
    collection: str,
    window_size: int,
    payload_map: dict[str, dict],
) -> RetrievedChunk:
    """Expand one chunk to include its neighbors from the same document.

    Queries Qdrant for points with the same doc_id and a chunk_index within
    [chunk_index - window_size, chunk_index + window_size].  Sorts neighbors
    by chunk_index and merges text in order around the anchor chunk.

    Args:
        chunk:       The anchor RetrievedChunk to expand.
        client:      Async Qdrant client.
        collection:  Collection name.
        window_size: Number of neighboring chunks on each side.
        payload_map: Pre-fetched {point_id: payload} for already-known points
                     (avoids a round trip for the anchor itself).

    Returns:
        New RetrievedChunk with expanded text.  Metadata (filename, page,
        score, id) is inherited from the original anchor chunk.
        Returns the original chunk unchanged if chunk_index is absent from
        the payload (legacy data) or if the Qdrant scroll fails.
    """
    anchor_payload = payload_map.get(chunk.id, {})
    chunk_index = anchor_payload.get("chunk_index")
    doc_id = anchor_payload.get("doc_id")

    if chunk_index is None or doc_id is None:
        # Legacy chunk — no positional info, return unchanged
        logger.debug(
            "sentence_window.skip_no_index",
            chunk_id=chunk.id,
        )
        return chunk

    chunk_index = int(chunk_index)
    low = max(0, chunk_index - window_size)
    high = chunk_index + window_size

    neighbor_filter = Filter(
        must=[
            FieldCondition(
                key="doc_id",
                match=MatchValue(value=doc_id),
            ),
            FieldCondition(
                key="chunk_index",
                range=Range(gte=low, lte=high),
            ),
        ]
    )

    try:
        scroll_result, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=neighbor_filter,
            limit=window_size * 2 + 1,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as exc:
        logger.warning(
            "sentence_window.scroll_failed",
            chunk_id=chunk.id,
            error=str(exc),
        )
        return chunk

    if not scroll_result:
        return chunk

    # Sort all neighbors (including anchor) by chunk_index
    neighbor_chunks: list[tuple[int, str, str]] = []  # (index, point_id, text)
    for point in scroll_result:
        p_payload = point.payload or {}
        p_index = p_payload.get("chunk_index")
        p_text = p_payload.get("text", "")
        p_id = str(point.id)
        if p_index is not None and p_text:
            neighbor_chunks.append((int(p_index), p_id, p_text))

    if not neighbor_chunks:
        return chunk

    neighbor_chunks.sort(key=lambda x: x[0])

    # Build merged text: mark anchor position with "[...]" separators
    parts: list[str] = []
    anchor_added = False
    for idx, p_id, p_text in neighbor_chunks:
        if idx == chunk_index:
            parts.append(p_text)
            anchor_added = True
        else:
            parts.append(p_text)

    if not anchor_added:
        # Anchor wasn't in scroll results; fall back to original text
        parts = [chunk.text]

    expanded_text = " [...] ".join(p for p in parts if p.strip())

    if not expanded_text.strip():
        return chunk

    logger.debug(
        "sentence_window.expanded",
        chunk_id=chunk.id,
        original_len=len(chunk.text),
        expanded_len=len(expanded_text),
        neighbors=len(neighbor_chunks),
    )

    return replace(chunk, text=expanded_text)


async def expand_chunks_to_window(
    chunks: list[RetrievedChunk],
    client: AsyncQdrantClient,
    collection: str,
    window_size: int = 2,
) -> list[RetrievedChunk]:
    """Expand a list of retrieved chunks to include their textual neighbors.

    For each chunk, fetches surrounding chunks from the same document
    (same doc_id, adjacent chunk_index values) and merges their text to
    provide richer context without changing the chunk's citation metadata.

    All expansions run in parallel via asyncio.gather.

    Args:
        chunks:      Retrieved chunks to expand (e.g. reranker output).
        client:      Async Qdrant client.
        collection:  Qdrant collection name.
        window_size: How many neighbors on each side to include (default 2).
                     window_size=2 → up to 5 chunks merged per anchor.

    Returns:
        List of RetrievedChunk objects in the same order as input.
        Each chunk's .text is replaced with the expanded window text.
        Chunks whose payload lacks chunk_index are returned unchanged.
    """
    if not chunks:
        return []

    log = logger.bind(
        collection=collection,
        window_size=window_size,
        num_chunks=len(chunks),
    )
    log.info("sentence_window.expand_start")

    # Fetch payloads for all anchor chunks in one batch scroll to get doc_id / chunk_index
    chunk_ids = [chunk.id for chunk in chunks]
    payload_map: dict[str, dict] = {}

    try:
        from qdrant_client.models import HasIdCondition  # type: ignore[attr-defined]
        id_filter = Filter(
            must=[HasIdCondition(has_id=chunk_ids)]
        )
        anchor_points, _ = await client.scroll(
            collection_name=collection,
            scroll_filter=id_filter,
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False,
        )
        for point in anchor_points:
            payload_map[str(point.id)] = point.payload or {}
    except Exception as exc:
        log.warning(
            "sentence_window.anchor_fetch_failed_falling_back",
            error=str(exc),
        )
        # Fall back: we have no anchor payloads — chunks will be returned unchanged
        # because chunk_index will be None for all of them

    # Expand all chunks in parallel
    tasks = [
        _expand_single_chunk(
            chunk=chunk,
            client=client,
            collection=collection,
            window_size=window_size,
            payload_map=payload_map,
        )
        for chunk in chunks
    ]

    results: list[RetrievedChunk] = []
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            log.warning(
                "sentence_window.single_expand_failed",
                chunk_id=chunks[i].id,
                error=str(result),
            )
            results.append(chunks[i])  # keep original on error
        else:
            results.append(result)

    expanded_count = sum(
        1 for orig, exp in zip(chunks, results)
        if exp.text != orig.text
    )
    log.info(
        "sentence_window.expand_done",
        expanded=expanded_count,
        unchanged=len(chunks) - expanded_count,
    )
    return results
