from __future__ import annotations

from typing import NamedTuple

import structlog
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from api.exceptions import EmbeddingError

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Model name used by Qdrant's native BM25 integration via fastembed.
_BM25_MODEL = "Qdrant/bm25"


class SparseEncoderResult(NamedTuple):
    """Sparse vector representation compatible with Qdrant's SparseVector.

    Attributes:
        indices: Non-zero dimension indices.
        values: Corresponding TF-IDF / BM25 weights.
    """

    indices: list[int]
    values: list[float]


class SparseEncoder:
    """BM25 sparse encoder backed by fastembed.

    Wraps ``fastembed.SparseTextEmbedding`` and converts its output to
    Qdrant-compatible :class:`SparseVector` objects.

    Note:
        fastembed's ``embed()`` returns a **generator**, not a list.
        Always call ``list()`` on its output — this class handles that
        internally.

    Example::

        encoder = SparseEncoder()
        sparse_vec = encoder.encode("what are transformer attention heads?")
        # SparseVector(indices=[...], values=[...])
    """

    def __init__(self) -> None:
        self._log = logger.bind(model=_BM25_MODEL)
        self._log.info("Loading BM25 sparse encoder")
        try:
            self._model = SparseTextEmbedding(model_name=_BM25_MODEL)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load fastembed BM25 model: {_BM25_MODEL}",
                detail=str(exc),
            ) from exc
        self._log.info("BM25 sparse encoder ready")

    def encode(self, text: str) -> SparseVector:
        """Encode a single text into a BM25 sparse vector.

        Args:
            text: Input string to encode.

        Returns:
            Qdrant :class:`SparseVector` with ``indices`` and ``values``.

        Raises:
            EmbeddingError: If fastembed raises during encoding.
        """
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        """Encode multiple texts into BM25 sparse vectors.

        Args:
            texts: List of strings to encode.

        Returns:
            List of :class:`SparseVector` in the same order as ``texts``.

        Raises:
            EmbeddingError: If fastembed raises during encoding.
        """
        if not texts:
            return []

        try:
            # fastembed returns a generator — must materialise with list().
            raw_embeddings = list(self._model.embed(texts))
        except Exception as exc:
            raise EmbeddingError(
                "fastembed BM25 encoding failed",
                detail=str(exc),
            ) from exc

        results: list[SparseVector] = []
        for emb in raw_embeddings:
            # fastembed SparseEmbedding has .indices and .values attributes.
            results.append(
                SparseVector(
                    indices=emb.indices.tolist(),
                    values=[float(v) for v in emb.values.tolist()],
                )
            )

        self._log.debug("Sparse encoding complete", count=len(results))
        return results
