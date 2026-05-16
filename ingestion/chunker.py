from __future__ import annotations

import structlog
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

from api.exceptions import ChunkingError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _table_to_text(rows: list[dict[str, str | None]]) -> str:
    """Serialise a table (list of row-dicts) to a markdown-style string.

    Args:
        rows: List of dicts mapping column name → cell value.

    Returns:
        Multi-line string with one row per line, suitable for embedding.
    """
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [" | ".join(headers)]
    lines.append(" | ".join(["---"] * len(headers)))
    for row in rows:
        lines.append(" | ".join(str(row.get(h, "")) or "" for h in headers))
    return "\n".join(lines)


def build_chunker() -> SemanticChunker:
    """Instantiate a SemanticChunker with project-standard settings.

    Uses ``text-embedding-3-large`` (same model as the index) so breakpoint
    distances are computed in the same embedding space.

    Returns:
        Configured :class:`SemanticChunker` instance.
    """
    settings = get_settings()
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )


def chunk_text(
    chunker: SemanticChunker,
    text: str,
    base_metadata: dict[str, object],
    *,
    min_chunk_size: int | None = None,
) -> list[dict[str, object]]:
    """Split a text string into semantic chunks with injected metadata.

    Args:
        chunker: Pre-built :class:`SemanticChunker` instance.
        text: Raw text to split.
        base_metadata: Metadata dict applied to every chunk (doc_id,
            filename, page_number, content_type, page_fingerprint, etc.).
        min_chunk_size: Minimum character length; chunks shorter than this
            are dropped. Defaults to ``settings.min_chunk_size``.

    Returns:
        List of dicts, each containing ``"text"`` and merged metadata fields.

    Raises:
        ChunkingError: If the chunker raises an unexpected exception.
    """
    settings = get_settings()
    threshold = min_chunk_size if min_chunk_size is not None else settings.min_chunk_size

    if not text or not text.strip():
        return []

    try:
        docs = chunker.create_documents([text])
    except Exception as exc:
        raise ChunkingError(
            "SemanticChunker failed to split text",
            detail=str(exc),
        ) from exc

    chunks: list[dict[str, object]] = []
    for idx, doc in enumerate(docs):
        chunk_text_str = doc.page_content.strip()
        if len(chunk_text_str) < threshold:
            logger.debug(
                "Dropping short chunk",
                chunk_index=idx,
                length=len(chunk_text_str),
                threshold=threshold,
            )
            continue
        chunk_meta: dict[str, object] = {
            **base_metadata,
            "chunk_index": idx,
            "text": chunk_text_str,
            "token_count": len(chunk_text_str.split()),  # rough token proxy
            "image_b64": None,
        }
        chunks.append(chunk_meta)

    return chunks


def chunk_table(
    rows: list[dict[str, str | None]],
    base_metadata: dict[str, object],
    chunk_index: int,
) -> dict[str, object] | None:
    """Convert a parsed table into a single indexable chunk.

    Tables are kept whole (not split further) because splitting mid-table
    destroys relational context. The table is serialised to markdown text.

    Args:
        rows: List of row-dicts from pdfplumber.
        base_metadata: Metadata applied to the chunk.
        chunk_index: Positional index within the page's chunk sequence.

    Returns:
        A chunk dict or ``None`` if the table serialises to an empty string.
    """
    settings = get_settings()
    text = _table_to_text(rows)
    if len(text) < settings.min_chunk_size:
        return None
    return {
        **base_metadata,
        "chunk_index": chunk_index,
        "text": text,
        "token_count": len(text.split()),
        "image_b64": None,
        "content_type": "table",
    }


def chunk_image(
    b64_str: str,
    ocr_text: str,
    content_type: str,
    base_metadata: dict[str, object],
    chunk_index: int,
) -> dict[str, object]:
    """Wrap an image/chart as a single chunk with its base64 payload.

    Args:
        b64_str: Base64-encoded image string.
        ocr_text: OCR'd text from the image (may be empty).
        content_type: ``"image"`` or ``"chart"``.
        base_metadata: Metadata applied to the chunk.
        chunk_index: Positional index within the page's chunk sequence.

    Returns:
        A chunk dict with ``image_b64`` populated.
    """
    fallback_text = ocr_text.strip() if ocr_text else f"[{content_type} on page {base_metadata.get('page_number')}]"
    return {
        **base_metadata,
        "chunk_index": chunk_index,
        "text": fallback_text,
        "token_count": len(fallback_text.split()),
        "image_b64": b64_str,
        "content_type": content_type,
    }
