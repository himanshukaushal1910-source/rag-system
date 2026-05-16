from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from qdrant_client.models import (
    Filter,
    FusionQuery,
    Prefetch,
    QueryRequest,
    SparseVector,
)

from api.exceptions import RetrievalError
from config import get_settings
from ingestion.embedder import AsyncEmbedder
from ingestion.sparse_encoder import SparseEncoder
from retrieval.qdrant_client import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantClientSingleton,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single chunk returned from hybrid retrieval.

    Attributes:
        chunk_id: Qdrant point ID (UUID5 string).
        score: Fusion score assigned by Qdrant RRF.
        doc_id: Source document UUID.
        filename: Original PDF filename.
        page_number: 1-indexed page number.
        chunk_index: Position within the page's chunk sequence.
        content_type: One of ``"text"``, ``"table"``, ``"image"``, ``"chart"``.
        text: Chunk text content (or OCR placeholder for images).
        image_b64: Base64 image string for visual chunks, else ``None``.
        token_count: Approximate token count of the chunk.
    """

    chunk_id: str
    score: float
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    content_type: str
    text: str
    image_b64: str | None = field(default=None)
    token_count: int = field(default=0)

    @property
    def citation(self) -> str:
        """Inline citation string for use in generated answers."""
        return f"[Doc: {self.filename}, Page: {self.page_number}]"


class HybridRetriever:
    """Hybrid retriever combining dense (OpenAI) and sparse (BM25) search.

    Uses Qdrant's native ``QueryRequest`` with ``prefetch`` + RRF fusion
    (Reciprocal Rank Fusion). RRF is handled server-side by Qdrant — no
    manual implementation needed.

    Architecture::

        query
          ├── dense embed (OpenAI) → Qdrant prefetch (dense vector)
          └── sparse encode (BM25) → Qdrant prefetch (sparse vector)
                            ↓
                    Qdrant RRF fusion
                            ↓
                    top-N fused results

    Args:
        embedder: Async OpenAI embedder instance.
        sparse_encoder: fastembed BM25 encoder instance.

    Example::

        retriever = HybridRetriever(embedder, sparse_encoder)
        chunks = await retriever.retrieve("what is grievance redressal?")
    """

    def __init__(
        self,
        embedder: AsyncEmbedder,
        sparse_encoder: SparseEncoder,
    ) -> None:
        self._embedder = embedder
        self._sparse_encoder = sparse_encoder
        self._settings = get_settings()
        self._log = logger.bind(component="HybridRetriever")

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        query_filter: Filter | None = None,
    ) -> list[RetrievedChunk]:
        """Run hybrid search and return fused, ranked chunks.

        Args:
            query: Natural language query string.
            top_k: Number of results to return. Defaults to
                ``settings.max_retrieval_chunks``.
            query_filter: Optional Qdrant :class:`Filter` for metadata
                filtering. Build with :func:`retrieval.filters.build_filter`.

        Returns:
            List of :class:`RetrievedChunk` ordered by RRF score descending.

        Raises:
            RetrievalError: If Qdrant search fails.
        """
        limit = top_k or self._settings.max_retrieval_chunks
        collection = self._settings.qdrant_collection_name
        log = self._log.bind(query=query[:80], limit=limit)

        log.info("Starting hybrid retrieval")

        # ------------------------------------------------------------------ #
        # 1. Encode query — dense and sparse in parallel
        # ------------------------------------------------------------------ #
        try:
            dense_vector = await self._embedder.embed_query(query)
        except Exception as exc:
            raise RetrievalError(
                "Dense query embedding failed", detail=str(exc)
            ) from exc

        try:
            sparse_vector: SparseVector = self._sparse_encoder.encode(query)
        except Exception as exc:
            raise RetrievalError(
                "Sparse query encoding failed", detail=str(exc)
            ) from exc

        log.debug(
            "Query encoded",
            dense_dim=len(dense_vector),
            sparse_nnz=len(sparse_vector.indices),
        )

        # ------------------------------------------------------------------ #
        # 2. Build Qdrant QueryRequest with prefetch + RRF fusion
        # ------------------------------------------------------------------ #
        # Each prefetch leg fetches 2× limit so RRF has enough candidates.
        prefetch_limit = limit * 2

        try:
            client = await QdrantClientSingleton.get()

            results = await client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            raise RetrievalError(
                "Qdrant hybrid query failed", detail=str(exc)
            ) from exc

        # ------------------------------------------------------------------ #
        # 3. Map results to RetrievedChunk dataclasses
        # ------------------------------------------------------------------ #
        chunks: list[RetrievedChunk] = []
        for point in results.points:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(point.id),
                    score=point.score,
                    doc_id=str(payload.get("doc_id", "")),
                    filename=str(payload.get("filename", "")),
                    page_number=int(payload.get("page_number", 0)),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    content_type=str(payload.get("content_type", "text")),
                    text=str(payload.get("text", "")),
                    image_b64=payload.get("image_b64"),
                    token_count=int(payload.get("token_count", 0)),
                )
            )

        log.info("Hybrid retrieval complete", results=len(chunks))
        return chunks
