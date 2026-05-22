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
    """Cross-encoder reranker using BAAI/bge-reranker-large.

    Upgraded from ms-marco-MiniLM-L-6-v2 to bge-reranker-large:
    - Trained on academic/scientific content (matches our PDF corpus)
    - +6.9 NDCG@10 improvement on BEIR benchmark
    - Runs on CUDA (GTX 1650 has 4.3GB VRAM, model needs ~2.2GB)
    - Still async-safe via run_in_executor

    BGE reranker difference vs ms-marco:
    - ms-marco outputs raw logits (any float range)
    - BGE outputs sigmoid-activated scores (0.0 to 1.0)
    - Both handled correctly — we just sort descending either way
    """

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.reranker_model
        self._top_k_final = settings.top_k_final
        self._log = logger.bind(model=self._model_name)

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log.info("Loading cross-encoder model", device=self._device, model=self._model_name)

        try:
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                model_kwargs={
                    "ignore_mismatched_sizes": True,
                },
            )
        except Exception as exc:
            raise RerankerError(
                f"Failed to load cross-encoder: {self._model_name}",
                detail=str(exc),
            ) from exc

        self._log.info("Cross-encoder model ready", device=self._device)

    def _predict_sync(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run synchronous cross-encoder inference."""
        scores = self._model.predict(
            pairs,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks using BGE cross-encoder.

        Runs CrossEncoder.predict() in threadpool executor — never blocks
        the async event loop during model inference.

        Args:
            query:  Original query string.
            chunks: Candidate chunks from hybrid retrieval.
            top_k:  Number of top chunks to return. None = use top_k_final.

        Returns:
            Reranked list of RetrievedChunk, best first, truncated to top_k.

        Raises:
            RerankerError: If cross-encoder inference fails.
        """
        # Use explicit None check so top_k=0 is honoured (not collapsed to default)
        limit = top_k if top_k is not None else self._top_k_final
        log = self._log.bind(
            query=query[:80],
            candidates=len(chunks),
            top_k=limit,
        )

        if not chunks:
            log.warning("No chunks to rerank")
            return []

        log.info("reranker.start")

        pairs = [(query, chunk.text) for chunk in chunks]

        try:
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None,
                partial(self._predict_sync, pairs),
            )
        except Exception as exc:
            raise RerankerError(
                "Cross-encoder inference failed",
                detail=str(exc),
            ) from exc

        # Sort by score descending
        scored_chunks = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        reranked: list[RetrievedChunk] = []
        for chunk, score in scored_chunks[:limit]:
            chunk.score = score
            reranked.append(chunk)

        log.info(
            "reranker.done",
            returned=len(reranked),
            top_score=round(reranked[0].score, 4) if reranked else None,
            bottom_score=round(reranked[-1].score, 4) if reranked else None,
        )
        return reranked
