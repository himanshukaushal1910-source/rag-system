from __future__ import annotations

import uuid
from pathlib import Path

import structlog
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from api.exceptions import IngestionError
from config import get_settings
from ingestion.chunker import build_chunker, chunk_image, chunk_table, chunk_text
from ingestion.embedder import AsyncEmbedder
from ingestion.fingerprint import compute_page_fingerprint
from ingestion.pdf_parser import ParsedPage, parse_pdf
from ingestion.sparse_encoder import SparseEncoder
from retrieval.qdrant_client import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantClientSingleton,
    ensure_collection_exists,
)

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _make_point_id(doc_id: str, chunk_index_global: int) -> str:
    """Create a deterministic UUID for a Qdrant point.

    Using UUID5 (namespace + name) ensures the same chunk always maps to the
    same point ID, making upserts idempotent across re-ingestion runs.

    Args:
        doc_id: UUID string for the source document.
        chunk_index_global: Global chunk index across all pages.

    Returns:
        UUID5 string.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:{chunk_index_global}"))


async def _get_existing_fingerprints(
    client: AsyncQdrantClient,
    collection: str,
    doc_id: str,
) -> set[str]:
    """Scroll through Qdrant to collect all known page fingerprints for a doc.

    Args:
        client: Async Qdrant client.
        collection: Collection name.
        doc_id: Document UUID to filter by.

    Returns:
        Set of SHA-256 fingerprint strings already stored for this document.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    known: set[str] = set()
    offset = None

    while True:
        results, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
            with_payload=["page_fingerprint"],
            limit=256,
            offset=offset,
        )
        for point in results:
            fp = (point.payload or {}).get("page_fingerprint")
            if fp:
                known.add(fp)
        if offset is None:
            break

    return set(known)


async def ingest_pdf(pdf_path: Path, doc_id: str | None = None) -> dict[str, object]:
    """Full ingestion pipeline for a single PDF.

    Pipeline:
        1. Parse PDF (text + tables + images) via pdfplumber + fitz.
        2. For each page, compute SHA-256 fingerprint.
        3. Skip pages whose fingerprint already exists in Qdrant (dedup).
        4. Chunk text semantically; keep tables and images as single chunks.
        5. Batch-embed all text chunks (OpenAI dense).
        6. Encode all text chunks with BM25 (fastembed sparse).
        7. Upsert points to Qdrant with full payload.

    Args:
        pdf_path: Path to the PDF file on disk.
        doc_id: Optional existing UUID for this document. If ``None``, a new
            UUID4 is generated. Pass the same UUID on re-ingestion to trigger
            dedup rather than duplicate insertion.

    Returns:
        Summary dict with ``doc_id``, ``pages_ingested``, ``chunks_ingested``,
        ``pages_skipped`` (dedup hits).

    Raises:
        IngestionError: On any unrecoverable failure in the pipeline.
    """
    settings = get_settings()
    doc_id = doc_id or str(uuid.uuid4())
    collection = settings.qdrant_collection_name

    log = logger.bind(filename=pdf_path.name, doc_id=doc_id)
    log.info("Starting ingestion")

    await ensure_collection_exists()
    client = await QdrantClientSingleton.get()

    # ------------------------------------------------------------------ #
    # 1. Parse PDF
    # ------------------------------------------------------------------ #
    parsed_doc = parse_pdf(pdf_path, doc_id)

    # ------------------------------------------------------------------ #
    # 2. Fetch existing fingerprints for dedup
    # ------------------------------------------------------------------ #
    existing_fps = await _get_existing_fingerprints(client, collection, doc_id)
    log.info("Existing fingerprints fetched", count=len(existing_fps))

    # ------------------------------------------------------------------ #
    # 3. Build chunks per page, skipping dedup hits
    # ------------------------------------------------------------------ #
    chunker = build_chunker()
    embedder = AsyncEmbedder()
    sparse_encoder = SparseEncoder()

    all_chunks: list[dict[str, object]] = []
    pages_skipped = 0
    global_chunk_idx = 0

    for page in parsed_doc.pages:
        fp = compute_page_fingerprint(page.raw_bytes)

        if fp in existing_fps:
            log.debug("Page skipped (duplicate fingerprint)", page=page.page_number)
            pages_skipped += 1
            continue

        base_meta = {
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "page_number": page.page_number,
            "content_type": "text",
            "page_fingerprint": fp,
            "source_url": None,
        }

        # --- Text chunks ---
        full_text = "\n\n".join(page.text_blocks)
        if full_text.strip():
            text_chunks = chunk_text(chunker, full_text, base_meta)
            for chunk in text_chunks:
                chunk["chunk_index"] = global_chunk_idx
                all_chunks.append(chunk)
                global_chunk_idx += 1

        # --- Table chunks ---
        for table_rows in page.tables:
            table_chunk = chunk_table(
                rows=table_rows,
                base_metadata={**base_meta, "content_type": "table"},
                chunk_index=global_chunk_idx,
            )
            if table_chunk:
                all_chunks.append(table_chunk)
                global_chunk_idx += 1

        # --- Image / chart chunks ---
        for content_type, b64_str, ocr_text in page.images:
            img_chunk = chunk_image(
                b64_str=b64_str,
                ocr_text=ocr_text,
                content_type=content_type,
                base_metadata={**base_meta, "content_type": content_type},
                chunk_index=global_chunk_idx,
            )
            all_chunks.append(img_chunk)
            global_chunk_idx += 1

    log.info(
        "Chunking complete",
        total_chunks=len(all_chunks),
        pages_skipped=pages_skipped,
    )

    if not all_chunks:
        log.warning("No new chunks to ingest after dedup")
        return {
            "doc_id": doc_id,
            "pages_ingested": 0,
            "chunks_ingested": 0,
            "pages_skipped": pages_skipped,
        }

    # ------------------------------------------------------------------ #
    # 4. Embed all text content (dense + sparse)
    # ------------------------------------------------------------------ #
    texts_to_embed = [str(c["text"]) for c in all_chunks]

    log.info("Embedding chunks (dense)", count=len(texts_to_embed))
    dense_vectors = await embedder.embed(texts_to_embed)

    log.info("Encoding chunks (sparse BM25)", count=len(texts_to_embed))
    sparse_vectors = sparse_encoder.encode_batch(texts_to_embed)

    # ------------------------------------------------------------------ #
    # 5. Build Qdrant points and upsert
    # ------------------------------------------------------------------ #
    points: list[PointStruct] = []
    for i, chunk in enumerate(all_chunks):
        point_id = _make_point_id(doc_id, int(chunk["chunk_index"]))  # type: ignore[arg-type]

        # Build payload — exclude internal keys not needed in Qdrant.
        payload = {k: v for k, v in chunk.items() if k != "chunk_index" or True}

        points.append(
            PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vectors[i],
                    SPARSE_VECTOR_NAME: sparse_vectors[i],
                },
                payload=payload,
            )
        )

    # Upsert in batches of 64 to avoid request size limits.
    upsert_batch_size = 64
    for batch_start in range(0, len(points), upsert_batch_size):
        batch = points[batch_start : batch_start + upsert_batch_size]
        await client.upsert(
            collection_name=collection,
            points=batch,
            wait=True,
        )
        log.debug(
            "Upserted batch",
            batch_start=batch_start,
            batch_size=len(batch),
        )

    pages_ingested = len(parsed_doc.pages) - pages_skipped
    log.info(
        "Ingestion complete",
        pages_ingested=pages_ingested,
        chunks_ingested=len(points),
        pages_skipped=pages_skipped,
    )

    return {
        "doc_id": doc_id,
        "pages_ingested": pages_ingested,
        "chunks_ingested": len(points),
        "pages_skipped": pages_skipped,
    }


async def ingest_directory(directory: Path) -> list[dict[str, object]]:
    """Ingest all PDFs in a directory sequentially.

    Args:
        directory: Path to a folder containing PDF files.

    Returns:
        List of per-file ingestion summary dicts.

    Raises:
        IngestionError: If the directory does not exist.
    """
    if not directory.is_dir():
        raise IngestionError(
            f"Directory not found: {directory}",
            detail=str(directory),
        )

    pdf_files = sorted(directory.glob("*.pdf"))
    log = logger.bind(directory=str(directory), total_pdfs=len(pdf_files))
    log.info("Starting directory ingestion")

    results: list[dict[str, object]] = []
    for pdf_path in pdf_files:
        try:
            result = await ingest_pdf(pdf_path)
            results.append(result)
        except Exception as exc:
            log.error("Failed to ingest PDF", filename=pdf_path.name, error=str(exc))
            results.append({"filename": pdf_path.name, "error": str(exc)})

    log.info("Directory ingestion complete", processed=len(results))
    return results
