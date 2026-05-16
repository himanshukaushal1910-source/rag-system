from __future__ import annotations

import asyncio
from typing import Any

import structlog
from openai import AsyncOpenAI, APIStatusError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from api.exceptions import EmbeddingError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Retry on transient OpenAI errors (rate-limit, timeout, 5xx).
_RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError)


def _build_retry_decorator() -> Any:
    return retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )


class AsyncEmbedder:
    """Async batched OpenAI embeddings client.

    Wraps the OpenAI ``/v1/embeddings`` endpoint with:
    - Automatic batching (respects the 2048-input hard cap).
    - Exponential-backoff retry on rate-limit / timeout errors.
    - structlog logging for every batch.

    Attributes:
        client: Underlying :class:`AsyncOpenAI` client.
        model: Embedding model name.
        batch_size: Max inputs per API call.

    Example::

        embedder = AsyncEmbedder()
        vectors = await embedder.embed(["sentence one", "sentence two"])
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.batch_size = settings.openai_embed_batch_size
        self._log = logger.bind(model=self.model, batch_size=self.batch_size)

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts.

        Args:
            texts: Batch of strings to embed. Length must be ≤ ``batch_size``.

        Returns:
            List of float vectors, one per input text.

        Raises:
            EmbeddingError: On non-retryable API failure.
        """
        decorated = _build_retry_decorator()(self._call_api)
        return await decorated(texts)

    async def _call_api(self, texts: list[str]) -> list[list[float]]:
        """Raw OpenAI API call — wrapped by retry decorator.

        Args:
            texts: Texts to embed.

        Returns:
            List of float vectors.

        Raises:
            EmbeddingError: On non-retryable API errors.
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            # Response data is sorted by index — safe to just extract in order.
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except _RETRY_EXCEPTIONS:
            raise  # Let tenacity handle these.
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

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed an arbitrary number of texts, batching internally.

        Args:
            texts: List of strings to embed (any length).

        Returns:
            List of float vectors in the same order as ``texts``.

        Raises:
            EmbeddingError: If any batch fails after retries.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        batches = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        self._log.info("Embedding texts", total=len(texts), batches=len(batches))

        for batch_idx, batch in enumerate(batches):
            self._log.debug("Embedding batch", batch_idx=batch_idx, size=len(batch))
            try:
                vectors = await self._embed_batch(batch)
                all_vectors.extend(vectors)
            except EmbeddingError:
                raise
            except Exception as exc:
                raise EmbeddingError(
                    f"Batch {batch_idx} failed unexpectedly",
                    detail=str(exc),
                ) from exc

        self._log.info("Embedding complete", total_vectors=len(all_vectors))
        return all_vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (convenience wrapper).

        Args:
            text: Query string.

        Returns:
            Single float vector.
        """
        vectors = await self.embed([text])
        return vectors[0]
