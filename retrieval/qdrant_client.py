from __future__ import annotations

import asyncio
from typing import ClassVar

import structlog
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseVectorParams,
    VectorParams,
)

from api.exceptions import QdrantConnectionError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Named vector keys — referenced in hybrid_retriever and ingestor.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantClientSingleton:
    """Async Qdrant client singleton with collection lifecycle management.

    Usage::

        client = await QdrantClientSingleton.get()
        await client.upsert(...)

    Call :meth:`close` during application shutdown (FastAPI lifespan).
    """

    _instance: ClassVar[AsyncQdrantClient | None] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get(cls) -> AsyncQdrantClient:
        """Return the shared :class:`AsyncQdrantClient`, creating it if needed.

        Returns:
            Initialised async Qdrant client.

        Raises:
            QdrantConnectionError: If the client cannot connect to Qdrant.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = await cls._create()
        return cls._instance

    @classmethod
    async def _create(cls) -> AsyncQdrantClient:
        settings = get_settings()
        log = logger.bind(qdrant_url=settings.qdrant_url)
        log.info("Creating async Qdrant client")
        try:
            client = AsyncQdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=30,
            )
            # Ping by listing collections.
            await client.get_collections()
            log.info("Qdrant client connected")
            return client
        except Exception as exc:
            raise QdrantConnectionError(
                f"Cannot connect to Qdrant at {settings.qdrant_url}",
                detail=str(exc),
            ) from exc

    @classmethod
    async def close(cls) -> None:
        """Close the underlying HTTP connection pool.

        Call this from FastAPI lifespan shutdown.
        """
        async with cls._lock:
            if cls._instance is not None:
                await cls._instance.close()
                cls._instance = None
                logger.info("Qdrant client closed")


async def ensure_collection_exists() -> None:
    """Create the Qdrant collection if it does not already exist.

    Collection schema:
    - ``dense``: 3072-dim Cosine named vector (OpenAI text-embedding-3-large).
    - ``sparse``: BM25 named sparse vector (fastembed).

    Payload indices created on:
    - ``doc_id`` (keyword) — for per-document filtering.
    - ``page_number`` (integer) — for page-range filtering.
    - ``content_type`` (keyword) — for modality filtering.

    Raises:
        QdrantConnectionError: If the client cannot reach Qdrant.
    """
    settings = get_settings()
    client = await QdrantClientSingleton.get()
    collection = settings.qdrant_collection_name
    log = logger.bind(collection=collection)

    existing = {c.name for c in (await client.get_collections()).collections}

    if collection in existing:
        log.info("Collection already exists — skipping creation")
        return

    log.info("Creating Qdrant collection")
    try:
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=settings.embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=False,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )

        # --- Payload indices for fast metadata filtering ---
        await client.create_payload_index(
            collection_name=collection,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=collection,
            field_name="page_number",
            field_schema=PayloadSchemaType.INTEGER,
        )
        await client.create_payload_index(
            collection_name=collection,
            field_name="content_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )

        log.info("Collection created with payload indices")
    except Exception as exc:
        raise QdrantConnectionError(
            f"Failed to create collection '{collection}'",
            detail=str(exc),
        ) from exc
