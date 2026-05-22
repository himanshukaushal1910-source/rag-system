from __future__ import annotations

"""
ingestion/embedder.py

Optimized AsyncEmbedder:
- batch_size=2048 (OpenAI paid tier max — fewer round trips)
- _EMBED_SEMAPHORE=15 — shared with chunker's RateLimitedEmbeddings
- Sequential batch execution — no thundering herd
- Exponential backoff with jitter on RateLimitError
- Singleton HTTP client with connection pool
"""

import asyncio
import time

import httpx
import structlog
from openai import AsyncOpenAI, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from api.exceptions import EmbeddingError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError)

# ── Singleton OpenAI client ───────────────────────────────────────────────────
_openai_client: AsyncOpenAI | None = None


def _get_openai_client() -> AsyncOpenAI:
    """Return shared AsyncOpenAI client with connection pool."""
    global _openai_client
    if _openai_client is None:
        settings = get_settings()
        _openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            max_retries=0,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=25,
                    keepalive_expiry=30,
                ),
                timeout=httpx.Timeout(60.0),
            ),
        )
    return _openai_client


# ── Shared semaphore — controls ALL OpenAI embedding calls ───────────────────
# This semaphore is imported by chunker.py's RateLimitedEmbeddings so that
# SemanticChunker's internal calls share the same rate limiting as our
# main embedding calls. Both go through this semaphore.
#
# semaphore=15:
#   Main embedding: batch_size=2048, so typically 1 call per PDF
#   SemanticChunker: 1 call per page × 23 pages = 23 calls per PDF
#   25 PDFs parallel: up to 25×23 = 575 queued calls, 15 active at once
#   Smooth throughput, no 429s on Tier 1 (1M TPM)
_EMBED_SEMAPHORE = asyncio.Semaphore(15)


class AsyncEmbedder:
    """Async batched OpenAI embeddings with shared rate limiting.

    Uses batch_size=2048 (OpenAI paid tier max) for fewer API calls.
    Shares _EMBED_SEMAPHORE with SemanticChunker's RateLimitedEmbeddings
    so all OpenAI embedding calls are rate-controlled together.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.client = _get_openai_client()
        self.model = settings.embedding_model
        # Use full 2048 batch size — paid tier supports it
        self.batch_size = settings.openai_embed_batch_size
        self._log = logger.bind(model=self.model, batch_size=self.batch_size)

    async def _call_api(
        self, texts: list[str], batch_idx: int = 0
    ) -> list[list[float]]:
        """Single API call with semaphore + tenacity retry.

        Args:
            texts:     Texts to embed.
            batch_idx: Index for logging.

        Returns:
            List of embedding vectors.
        """
        @retry(
            retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
            wait=wait_exponential_jitter(initial=4, max=120, jitter=5),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        async def _inner() -> list[list[float]]:
            async with _EMBED_SEMAPHORE:
                t0 = time.monotonic()
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=texts,
                    )
                    elapsed = round(time.monotonic() - t0, 2)
                    self._log.debug(
                        "embedder.batch_done",
                        batch_idx=batch_idx,
                        size=len(texts),
                        elapsed_s=elapsed,
                    )
                    return [
                        item.embedding
                        for item in sorted(response.data, key=lambda x: x.index)
                    ]
                except RateLimitError:
                    self._log.warning(
                        "embedder.rate_limited",
                        batch_idx=batch_idx,
                        size=len(texts),
                    )
                    print(
                        f"\n⚠️  Rate limit hit (batch {batch_idx}) — "
                        f"retrying with backoff...",
                        flush=True,
                    )
                    raise
                except APITimeoutError:
                    self._log.warning("embedder.timeout", batch_idx=batch_idx)
                    print(
                        f"\n⚠️  API timeout (batch {batch_idx}) — retrying...",
                        flush=True,
                    )
                    raise
                except APIStatusError as exc:
                    raise EmbeddingError(
                        f"OpenAI Embeddings API error: {exc.status_code}",
                        detail=str(exc),
                    ) from exc
                except Exception as exc:
                    raise EmbeddingError(
                        "Unexpected error from OpenAI Embeddings API",
                        detail=str(exc),
                    ) from exc

        return await _inner()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with sequential batching.

        Sequential (not gather) to avoid thundering herd.
        Semaphore controls concurrency across all parallel PDFs.

        Args:
            texts: Strings to embed.

        Returns:
            Float vectors in same order as input.
        """
        if not texts:
            return []

        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        self._log.info(
            "embedder.start",
            total_texts=len(texts),
            num_batches=len(batches),
        )

        vectors: list[list[float]] = []
        for idx, batch in enumerate(batches):
            result = await self._call_api(batch, batch_idx=idx)
            vectors.extend(result)

        self._log.info("embedder.complete", total_vectors=len(vectors))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query string.

        Returns:
            Single embedding vector.
        """
        return (await self.embed([text]))[0]
