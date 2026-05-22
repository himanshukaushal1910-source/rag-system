from __future__ import annotations

"""
Optimized qdrant_client.py:

- Collection existence cached after first check — no repeated API calls
- invalidate_collection_cache() for explicit cache busting after deletion
- Singleton pattern with asyncio.Lock for thread safety
"""

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

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class QdrantClientSingleton:
    """Async Qdrant client singleton with connection pooling."""

    _instance: ClassVar[AsyncQdrantClient | None] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    async def get(cls) -> AsyncQdrantClient:
        """Return shared AsyncQdrantClient, creating if needed."""
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
                prefer_grpc=settings.use_grpc,
                grpc_port=settings.qdrant_grpc_port,
            )
            await client.get_collections()
            log.info(
                "Qdrant client connected",
                grpc=settings.use_grpc,
                grpc_port=settings.qdrant_grpc_port,
            )
            return client
        except Exception as exc:
            raise QdrantConnectionError(
                f"Cannot connect to Qdrant at {settings.qdrant_url}",
                detail=str(exc),
            ) from exc

    @classmethod
    async def close(cls) -> None:
        """Close connection pool — call from FastAPI lifespan shutdown."""
        async with cls._lock:
            if cls._instance is not None:
                await cls._instance.close()
                cls._instance = None
                logger.info("Qdrant client closed")


# Cache collection existence — avoid repeated get_collections() API calls
# Safe because collections are never deleted during normal operation
_collection_exists_cache: set[str] = set()


async def ensure_collection_exists(collection_name: str | None = None) -> None:
    """Create Qdrant collection if it doesn't exist.

    Caches result after first successful check — subsequent calls
    return instantly without hitting Qdrant API.

    Args:
        collection_name: Override collection name. Defaults to settings value.
    """
    settings = get_settings()
    collection = collection_name or settings.qdrant_collection_name
    log = logger.bind(collection=collection)

    # Return instantly if already confirmed to exist
    if collection in _collection_exists_cache:
        return

    client = await QdrantClientSingleton.get()
    existing = {c.name for c in (await client.get_collections()).collections}

    if collection in existing:
        log.info("Collection already exists — skipping creation")
        _collection_exists_cache.add(collection)
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

        # Payload indices for fast metadata filtering
        for field_name, schema in [
            ("doc_id", PayloadSchemaType.KEYWORD),
            ("filename", PayloadSchemaType.KEYWORD),
            ("content_type", PayloadSchemaType.KEYWORD),
            ("page_number", PayloadSchemaType.INTEGER),
        ]:
            await client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=schema,
            )

        log.info("Collection created with payload indices")
        _collection_exists_cache.add(collection)

    except Exception as exc:
        raise QdrantConnectionError(
            f"Failed to create collection '{collection}'",
            detail=str(exc),
        ) from exc


def invalidate_collection_cache(collection_name: str | None = None) -> None:
    """Remove collection from existence cache.

    Call after explicitly deleting a collection so next
    ensure_collection_exists() re-checks Qdrant.

    Args:
        collection_name: Collection to invalidate. None clears all.
    """
    if collection_name is None:
        _collection_exists_cache.clear()
        logger.info("Collection cache cleared")
    else:
        _collection_exists_cache.discard(collection_name)
        logger.info("Collection cache invalidated", collection=collection_name)
