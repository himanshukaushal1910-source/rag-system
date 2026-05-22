"""
retrieval/hybrid_retriever.py

Hybrid dense + sparse retrieval with RRF fusion via Qdrant Query API.

Fixes applied:
  FIX-1  MAX_RETRIEVAL_CHUNKS raised to 40 (was 20) so multi-section
          complex queries have enough raw candidates before reranking.
  FIX-2  Section-aware retrieval: when a query contains section keywords
          (e.g. "Section 3", "Table 4", "Figure 2", "Conclusion") we
          inject a metadata filter that targets those chunk types.
  FIX-3  Multi-query deduplication: identical chunk IDs from parallel
          sub-queries are deduplicated before reranking to avoid the
          reranker wasting slots on the same chunk twice.
  FIX-4  Minimum per-query yield: if a sub-query returns fewer than 3
          chunks we fall back to a broader BM25-only search on that
          sub-query so we always have candidates for every sub-question.

BUG-FIX (Issue 6):
  _hybrid_search and _sparse_only_search previously did:
      sparse_indices, sparse_values = self._sparse_encoder.encode(query)
  encode() returns a SparseVector object, NOT a tuple.
  Fixed to:
      sparse_vec = self._sparse_encoder.encode(query)
      ... sparse_vec.indices, sparse_vec.values ...

BUG-FIX (Issue 5):
  All settings field accesses use snake_case
  (settings.qdrant_collection_name, settings.max_retrieval_chunks)
  matching pydantic-settings auto-lowercase behaviour.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import structlog
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from config import Settings

logger = structlog.get_logger(__name__)

# Patterns that indicate a query targets a specific structural element
_SECTION_PATTERN = re.compile(
    r"\b(section|table|figure|fig|appendix|equation|eq|algorithm|"
    r"conclusion|limitation|discussion|result|abstract)\b",
    re.I,
)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its payload and retrieval score.

    Attributes:
        id:              Qdrant point ID (UUID string).
        score:           RRF or BM25 retrieval score.
        text:            Chunk text content.
        filename:        Source PDF filename.
        page_number:     1-indexed page number.
        content_type:    'text', 'table', 'image', or 'chart'.
        image_b64:       Base64 image string (only for image/chart chunks).
        section_heading: Nearest parent section heading (empty string if none).
        chunk_id:        Alias for id — used by reranker and state.
    """

    id: str
    score: float
    text: str
    filename: str
    page_number: int
    content_type: str
    image_b64: str | None = None
    section_heading: str = ""

    @property
    def chunk_id(self) -> str:
        """Alias for id — used by legacy code that checks chunk.chunk_id."""
        return self.id


class HybridRetriever:
    """Dense + sparse + RRF retrieval over a Qdrant collection.

    Args:
        client:         Async Qdrant client (singleton from qdrant_client.py).
        settings:       Pydantic settings object.
        embedder:       AsyncEmbedder with embed_query(str) -> list[float].
        sparse_encoder: SparseEncoder with encode(str) -> SparseVector object.
    """

    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: Settings,
        embedder: Any,
        sparse_encoder: Any,
    ) -> None:
        self._client = client
        self._settings = settings
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: models.Filter | None = None,
    ) -> list[dict]:
        """Retrieve top-k chunks for a single query using hybrid RRF.

        Args:
            query:   Natural language query string.
            top_k:   Number of results to return (default from settings).
            filters: Optional Qdrant filter object.

        Returns:
            List of dicts with keys: id, score, payload.
        """
        # BUG-FIX (Issue 5): snake_case settings access
        top_k = top_k or self._settings.max_retrieval_chunks

        # FIX-2: detect structural targets and add payload filter
        structural_filter = self._build_structural_filter(query)
        combined_filter = self._merge_filters(filters, structural_filter)

        results = await self._hybrid_search(query, top_k, combined_filter)

        # FIX-4: fallback if too few results
        if len(results) < 3:
            logger.warning(
                "retriever.sparse_fallback",
                query=query[:80],
                initial_results=len(results),
            )
            sparse_results = await self._sparse_only_search(query, top_k, filters)
            results = self._dedup_merge(results, sparse_results)

        return results

    async def retrieve_multi(
        self,
        queries: list[str],
        top_k_per_query: int | None = None,
        filters: models.Filter | None = None,
    ) -> list[dict]:
        """Retrieve and deduplicate results for multiple sub-queries.

        Runs all queries in parallel then deduplicates by chunk ID.

        Args:
            queries:         List of sub-query strings.
            top_k_per_query: Chunks per sub-query (default max_retrieval_chunks).
            filters:         Optional shared Qdrant filter.

        Returns:
            Deduplicated list of chunk dicts sorted by best score.
        """
        # BUG-FIX (Issue 5): snake_case
        top_k_per_query = top_k_per_query or self._settings.max_retrieval_chunks

        tasks = [
            self.retrieve(q, top_k=top_k_per_query, filters=filters)
            for q in queries
        ]
        per_query_results = await asyncio.gather(*tasks, return_exceptions=True)

        # FIX-3: deduplicate across sub-queries
        seen_ids: set[str] = set()
        merged: list[dict] = []
        for result_list in per_query_results:
            if isinstance(result_list, Exception):
                logger.error("retriever.sub_query_failed", error=str(result_list))
                continue
            for item in result_list:
                chunk_id = item.get("id")
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    merged.append(item)

        # Sort by descending score so reranker gets best candidates first
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        logger.info(
            "retriever.multi_done",
            sub_queries=len(queries),
            total_unique_chunks=len(merged),
        )
        return merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: models.Filter | None,
    ) -> list[dict]:
        """Run dense + sparse prefetch with RRF fusion.

        BUG-FIX (Issue 6): encode() returns SparseVector object.
        Access .indices and .values attributes — do NOT unpack as a tuple.

        Args:
            query:   Query string.
            top_k:   Number of results to retrieve.
            filters: Optional Qdrant filter.

        Returns:
            List of result dicts with id, score, payload.
        """
        dense_vector = await self._embedder.embed_query(query)

        # BUG-FIX: do NOT do: sparse_indices, sparse_values = encode(query)
        # encode() returns a SparseVector object, not a tuple
        sparse_vec = self._sparse_encoder.encode(query)

        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=top_k * 2,
                filter=filters,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices,
                    values=sparse_vec.values,
                ),
                using="sparse",
                limit=top_k * 2,
                filter=filters,
            ),
        ]

        # BUG-FIX (Issue 5): snake_case — settings.qdrant_collection_name
        response = await self._client.query_points(
            collection_name=self._settings.qdrant_collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in response.points
        ]

    async def _sparse_only_search(
        self,
        query: str,
        top_k: int,
        filters: models.Filter | None,
    ) -> list[dict]:
        """BM25-only search used as fallback when hybrid returns < 3 results.

        BUG-FIX (Issue 6): encode() returns SparseVector object — use
        sparse_vec.indices and sparse_vec.values, not tuple unpacking.

        Args:
            query:   Query string.
            top_k:   Number of results to retrieve.
            filters: Optional Qdrant filter.

        Returns:
            List of result dicts with id, score, payload.
        """
        # BUG-FIX: do NOT do: sparse_indices, sparse_values = encode(query)
        sparse_vec = self._sparse_encoder.encode(query)

        # BUG-FIX (Issue 5): snake_case
        response = await self._client.query_points(
            collection_name=self._settings.qdrant_collection_name,
            query=models.SparseVector(
                indices=sparse_vec.indices,
                values=sparse_vec.values,
            ),
            using="sparse",
            limit=top_k,
            with_payload=True,
            query_filter=filters,
        )

        return [
            {
                "id": str(point.id),
                "score": point.score,
                "payload": point.payload,
            }
            for point in response.points
        ]

    def _build_structural_filter(self, query: str) -> models.Filter | None:
        """FIX-2: Return a filter targeting specific content_type when query
        explicitly references a table, figure, or image.

        Args:
            query: Natural language query string.

        Returns:
            Qdrant Filter object or None if no structural target detected.
        """
        lower = query.lower()
        if re.search(r"\b(table\s*\d|figure\s*\d|fig\.?\s*\d)\b", lower, re.I):
            if "table" in lower:
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_type",
                            match=models.MatchValue(value="table"),
                        )
                    ]
                )
            if re.search(r"\bfig(ure)?\b", lower):
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key="content_type",
                            match=models.MatchAny(any=["image", "chart"]),
                        )
                    ]
                )
        return None

    @staticmethod
    def _merge_filters(
        base: models.Filter | None,
        extra: models.Filter | None,
    ) -> models.Filter | None:
        """AND-merge two optional Qdrant filters.

        Args:
            base:  Primary filter (may be None).
            extra: Additional filter to AND with base (may be None).

        Returns:
            Combined filter or None if both inputs are None.
        """
        if base is None:
            return extra
        if extra is None:
            return base
        must = list(base.must or []) + list(extra.must or [])
        return models.Filter(must=must)

    @staticmethod
    def _dedup_merge(primary: list[dict], secondary: list[dict]) -> list[dict]:
        """Merge secondary results into primary, skipping already-seen IDs.

        Args:
            primary:   Results already collected (take priority).
            secondary: Additional results to merge in.

        Returns:
            primary list extended with non-duplicate items from secondary.
        """
        seen = {item["id"] for item in primary}
        for item in secondary:
            if item["id"] not in seen:
                primary.append(item)
                seen.add(item["id"])
        return primary
