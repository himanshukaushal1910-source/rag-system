from __future__ import annotations

import asyncio
from functools import partial

import structlog
from sentence_transformers import CrossEncoder

from api.exceptions import RerankerError
from config import get_settings
from retrieval.hybrid_retriever import RetrievedChunk

logger: structlog.BoundLogger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """Cross-encoder re-ranker for retrieved chunks.

    Wraps ``sentence_transformers.CrossEncoder`` — which is synchronous —
    and runs inference in a threadpool executor so it never blocks the async
    event loop.

    The model should be loaded **once** at application startup (FastAPI
    lifespan) and stored on ``app.state``. Do not instantiate per-request.

    Args:
        model_name: HuggingFace model identifier. Defaults to
            ``settings.reranker_model``.

    Example::

        reranker = CrossEncoderReranker()
        reranked = await reranker.rerank(query, chunks, top_k=5)
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model
        self._top_k_final = settings.top_k_final
        self._log = logger.bind(model=self._model_name)

        self._log.info("Loading cross-encoder model")
        try:
            self._model = CrossEncoder(self._model_name)
        except Exception as exc:
            raise RerankerError(
                f"Failed to load cross-encoder: {self._model_name}",
                detail=str(exc),
            ) from exc
        self._log.info("Cross-encoder model ready")

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run synchronous cross-encoder inference.

        Args:
            pairs: List of (query, chunk_text) pairs.

        Returns:
            List of float relevance scores.
        """
        scores = self._model.predict(pairs)
        # CrossEncoder returns numpy float32 array — cast to Python floats.
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Re-rank retrieved chunks using the cross-encoder.

        Runs ``CrossEncoder.predict()`` in a threadpool executor to avoid
        blocking the event loop during model inference.

        Args:
            query: Original query string.
            chunks: Candidate chunks from hybrid retrieval.
            top_k: Number of top chunks to return after re-ranking.
                Defaults to ``settings.top_k_final``.

        Returns:
            Re-ranked list of :class:`RetrievedChunk`, best first, truncated
            to ``top_k``.

        Raises:
            RerankerError: If cross-encoder inference fails.
        """
        limit = top_k or self._top_k_final
        log = self._log.bind(query=query[:80], candidates=len(chunks), top_k=limit)

        if not chunks:
            log.warning("No chunks to rerank")
            return []

        log.info("Starting cross-encoder reranking")

        # Build (query, text) pairs for the cross-encoder.
        pairs = [(query, chunk.text) for chunk in chunks]

        # Run blocking inference in threadpool — never blocks the event loop.
        try:
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                partial(self._predict_sync, pairs),
            )
        except Exception as exc:
            raise RerankerError(
                "Cross-encoder inference failed",
                detail=str(exc),
            ) from exc

        # Attach scores and sort descending.
        scored_chunks = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Return top_k chunks with updated scores.
        reranked: list[RetrievedChunk] = []
        for chunk, score in scored_chunks[:limit]:
            # Replace RRF fusion score with cross-encoder score.
            chunk.score = score
            reranked.append(chunk)

        log.info(
            "Reranking complete",
            returned=len(reranked),
            top_score=round(reranked[0].score, 4) if reranked else None,
        )
        return reranked
