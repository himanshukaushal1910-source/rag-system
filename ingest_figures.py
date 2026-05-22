"""
ingest_figures.py

Figure-only ingestion — adds GPT-4o figure description chunks to
existing Qdrant collection WITHOUT re-ingesting text/tables.

Fixes applied:
  C-5  get_running_loop() instead of deprecated get_event_loop()
  H-2  PDF directory from config (never hardcoded)
  L-7  chunk_index uses a globally-unique per-doc counter (not page_idx)
  L-9  Separate semaphore for vision API calls (not shared with embedder)

This script:
1. Scans each PDF for pages that likely have figures
2. Checks if a chart chunk already exists for that page in Qdrant
3. If not → renders page → GPT-4o vision → stores description chunk
4. Skips pages that already have chart chunks (idempotent)

Usage:
    python ingest_figures.py           # all PDFs in configured pdf_dir
    python ingest_figures.py 10        # max 10 concurrent PDFs
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

import fitz
import structlog
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from config import get_settings
from ingestion.embedder import AsyncEmbedder
from ingestion.figure_describer import (
    describe_figure_pages_batch,
    _page_likely_has_figure,
)
from ingestion.fingerprint import compute_page_fingerprint
from ingestion.sparse_encoder import SparseEncoder
from retrieval.qdrant_client import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    QdrantClientSingleton,
    ensure_collection_exists,
)

logger = structlog.get_logger(__name__)
console = Console()

# L-9: dedicated semaphore for GPT-4o vision calls — separate from embedding
# Vision calls are slow (~3-5s each) and expensive; limit concurrency tightly
_VISION_SEMAPHORE = asyncio.Semaphore(3)

# Semaphore for Qdrant upserts
_QDRANT_SEM = asyncio.Semaphore(5)


def _pdf_dir() -> Path:
    """Return PDF directory from config (never hardcoded)."""
    return Path(get_settings().pdf_dir)


def _make_point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}:fig:{chunk_index}"))


async def _get_existing_chart_pages(
    client,
    collection: str,
    doc_id: str,
) -> set[int]:
    """Get page numbers that already have chart chunks for this doc."""
    known: set[int] = set()
    offset = None

    while True:
        results, offset = await client.scroll(
            collection_name=collection,
            scroll_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="content_type", match=MatchValue(value="chart")),
            ]),
            with_payload=["page_number"],
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        for point in results:
            pn = (point.payload or {}).get("page_number")
            if pn:
                known.add(pn)
        if offset is None:
            break

    return known


async def ingest_figures_for_pdf(
    pdf_path: Path,
    embedder: AsyncEmbedder,
    sparse_encoder: SparseEncoder,
    openai_client: object,
    settings: object,
    progress: Progress,
    task_id: int,
) -> dict:
    """Add figure description chunks for one PDF."""
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, pdf_path.name))
    collection = settings.qdrant_collection_name
    log = logger.bind(filename=pdf_path.name)

    client = await QdrantClientSingleton.get()

    existing_chart_pages = await _get_existing_chart_pages(
        client, collection, doc_id
    )

    fitz_doc = fitz.open(str(pdf_path))
    total_pages = len(fitz_doc)

    page_texts: dict[int, str] = {}
    page_has_images: dict[int, bool] = {}

    for page_idx in range(total_pages):
        page_number = page_idx + 1
        if page_number in existing_chart_pages:
            continue
        page = fitz_doc[page_idx]
        text = page.get_text("text")
        blocks = page.get_text("dict")["blocks"]
        has_imgs = any(b.get("type") == 1 for b in blocks)
        page_texts[page_idx] = text
        page_has_images[page_idx] = has_imgs

    candidate_indices = [
        idx for idx in page_texts
        if _page_likely_has_figure(page_texts[idx], page_has_images.get(idx, False))
    ]

    if not candidate_indices:
        fitz_doc.close()
        progress.advance(task_id)
        return {"filename": pdf_path.name, "figures_added": 0, "skipped": True}

    log.info("figure_ingest.start", candidates=len(candidate_indices))

    filtered_texts = {i: page_texts[i] for i in candidate_indices}
    filtered_imgs = {i: page_has_images.get(i, False) for i in candidate_indices}

    # L-9: use dedicated vision semaphore instead of embedding semaphore
    descriptions = await describe_figure_pages_batch(
        fitz_doc=fitz_doc,
        page_texts=filtered_texts,
        page_has_images=filtered_imgs,
        filename=pdf_path.name,
        openai_client=openai_client,
        semaphore=_VISION_SEMAPHORE,
        model=settings.figure_description_model,
        max_per_doc=settings.figure_description_max_per_doc,
    )
    fitz_doc.close()

    if not descriptions:
        progress.advance(task_id)
        return {"filename": pdf_path.name, "figures_added": 0}

    # Build chunks — L-7: use enumerated counter for globally-unique chunk_index
    chunks: list[dict] = []
    for chunk_counter, (page_idx, description) in enumerate(descriptions.items()):
        page_number = page_idx + 1
        page_text_raw = page_texts.get(page_idx, "")
        fingerprint = compute_page_fingerprint(
            page_text_raw.encode("utf-8", errors="replace")
        )
        prefix = f"[Document: {pdf_path.stem} | Section: Figure | Type: chart]\n"
        chunks.append({
            "doc_id": doc_id,
            "filename": pdf_path.name,
            "page_number": page_number,
            "chunk_index": chunk_counter,   # L-7: sequential, not page_idx
            "content_type": "chart",
            "text": prefix + description,
            "image_b64": None,
            "page_fingerprint": fingerprint,
            "token_count": len(description.split()),
            "source_url": None,
            "section_heading": "Figure",
        })

    texts = [c["text"] for c in chunks]
    loop = asyncio.get_running_loop()  # C-5: get_running_loop not get_event_loop
    dense_future = embedder.embed(texts)
    sparse_future = loop.run_in_executor(None, sparse_encoder.encode_batch, texts)
    dense_vectors, sparse_vectors = await asyncio.gather(dense_future, sparse_future)

    points = [
        PointStruct(
            id=_make_point_id(doc_id, int(c["chunk_index"])),
            vector={
                DENSE_VECTOR_NAME: dense_vectors[i],
                SPARSE_VECTOR_NAME: sparse_vectors[i],
            },
            payload=c,
        )
        for i, c in enumerate(chunks)
    ]

    async with _QDRANT_SEM:
        await client.upsert(
            collection_name=collection,
            points=points,
            wait=True,  # H-7: wait for durability
        )

    progress.advance(task_id)
    log.info("figure_ingest.done", figures_added=len(points))
    return {"filename": pdf_path.name, "figures_added": len(points)}


async def run(batch_size: int = 10) -> None:
    """Run figure-only ingestion on all PDFs."""
    settings = get_settings()

    if not settings.figure_description_enabled:
        console.print("[yellow]figure_description_enabled=False in config. Exiting.[/]")
        return

    await ensure_collection_exists()

    console.print("\n[bold cyan]Loading components...[/]")
    embedder = AsyncEmbedder()
    sparse_encoder = SparseEncoder()

    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

    pdf_dir = _pdf_dir()
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    total = len(pdfs)
    console.print(f"[bold]Found {total} PDFs[/] in {pdf_dir} — figure description only\n")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=4,
    )

    total_figures = 0
    failed = 0
    lock = asyncio.Lock()
    t0 = time.time()

    with progress:
        task = progress.add_task("[cyan]Processing PDFs...", total=total)

        async def process_one(pdf: Path) -> None:
            nonlocal total_figures, failed
            try:
                result = await ingest_figures_for_pdf(
                    pdf_path=pdf,
                    embedder=embedder,
                    sparse_encoder=sparse_encoder,
                    openai_client=openai_client,
                    settings=settings,
                    progress=progress,
                    task_id=task,
                )
                async with lock:
                    total_figures += result.get("figures_added", 0)
                    elapsed = time.time() - t0
                    rate = round(total_figures / max(elapsed, 1) * 60, 1)
                    progress.update(
                        task,
                        description=(
                            f"[cyan]figures: {total_figures} | "
                            f"fail: [red]{failed}[/] | "
                            f"{rate} figs/min"
                        ),
                    )
            except Exception as exc:
                logger.error("figure_ingest.pdf_failed", pdf=pdf.name, error=str(exc))
                async with lock:
                    failed += 1
                progress.advance(task)

        for i in range(0, total, batch_size):
            batch = pdfs[i: i + batch_size]
            await asyncio.gather(*[process_one(pdf) for pdf in batch])

    total_time = round((time.time() - t0) / 60, 1)
    console.print(f"\n[bold green]Done[/] — {total_figures} figure chunks added in {total_time} min")
    console.print(f"Failed: [red]{failed}[/]")


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    asyncio.run(run(batch_size=batch))
