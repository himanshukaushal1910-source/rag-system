"""
Main ingestion script — fully optimized parallel PDF ingestion.

Features:
- Components loaded once, shared across all PDFs
- Rich progress bar updating after EACH PDF (not each batch)
- Resume-safe (deterministic doc_id via uuid5)
- Configurable batch size via CLI
- Live ETA, chunk count, failure tracking

Usage:
    python ingest.py           # default batch=20
    python ingest.py 30        # batch=30
    python ingest.py 50        # batch=50
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel
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
from rich.table import Table

from config import get_settings
from ingestion.chunker import build_chunker
from ingestion.embedder import AsyncEmbedder
from ingestion.ingestor import ingest_pdf
from ingestion.sparse_encoder import SparseEncoder
from retrieval.qdrant_client import ensure_collection_exists

logger = structlog.get_logger(__name__)
console = Console()

PDF_DIR = Path(r"D:\rag_system\data\pdfs\papers")


async def run(batch_size: int = 20) -> None:
    """Run parallel PDF ingestion with per-PDF progress updates."""

    settings = get_settings()
    await ensure_collection_exists()

    console.print("\n[bold cyan]Loading components (one-time)...[/]")
    chunker = build_chunker(settings)
    embedder = AsyncEmbedder()
    sparse_encoder = SparseEncoder()
    console.print("[green]✓[/] Components ready\n")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    total = len(pdfs)
    console.print(f"[bold]Found {total} PDFs[/] in {PDF_DIR}")
    console.print(f"[bold]Batch size:[/] {batch_size}\n")

    # ── Progress bar ───────────────────────────────────────────────────────
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

    total_chunks = 0
    failed = 0
    skipped_pdfs = 0
    done = 0
    t0 = time.time()

    # Shared lock for updating counters from concurrent coroutines
    lock = asyncio.Lock()

    with progress:
        task = progress.add_task("[cyan]Ingesting PDFs...", total=total)

        async def ingest_one(pdf: Path) -> None:
            """Ingest a single PDF and update progress immediately on completion."""
            nonlocal total_chunks, failed, skipped_pdfs, done

            try:
                result = await ingest_pdf(
                    pdf,
                    chunker=chunker,
                    embedder=embedder,
                    sparse_encoder=sparse_encoder,
                )
                chunks = result.get("chunks_ingested", 0)
                pages_in = result.get("pages_ingested", 0)
                pages_sk = result.get("pages_skipped", 0)

                async with lock:
                    total_chunks += chunks
                    if pages_in == 0 and pages_sk > 0:
                        skipped_pdfs += 1
                    done += 1

            except Exception as exc:
                logger.error(
                    "PDF failed",
                    filename=pdf.name,
                    error=str(exc)[:120],
                )
                async with lock:
                    failed += 1
                    done += 1

            # ── Update progress bar immediately after this PDF ─────────
            elapsed = time.time() - t0
            rate = round(done / elapsed * 60, 1) if elapsed > 0 else 0
            success = done - failed - skipped_pdfs

            progress.update(
                task,
                advance=1,
                description=(
                    f"[cyan]chunks: {total_chunks:,} | "
                    f"ok: [green]{success}[/] | "
                    f"skip: [yellow]{skipped_pdfs}[/] | "
                    f"fail: [red]{failed}[/] | "
                    f"{rate} PDFs/min"
                ),
            )

        # ── Process in batches — all PDFs in a batch run concurrently ─────
        for i in range(0, total, batch_size):
            batch = pdfs[i : i + batch_size]
            await asyncio.gather(*[ingest_one(pdf) for pdf in batch])

    # ── Final summary ──────────────────────────────────────────────────────
    total_time = round((time.time() - t0) / 60, 1)
    success_count = total - failed - skipped_pdfs
    rate_final = round(total / total_time, 1) if total_time > 0 else 0

    console.print()
    console.print(Panel(
        f"[bold green]✓ Ingestion Complete[/]\n\n"
        f"  Total PDFs    : {total}\n"
        f"  Successful    : [green]{success_count}[/]\n"
        f"  Skipped       : [yellow]{skipped_pdfs}[/] (already done)\n"
        f"  Failed        : [red]{failed}[/]\n"
        f"  Total chunks  : [cyan]{total_chunks:,}[/]\n"
        f"  Total time    : {total_time} minutes\n"
        f"  Rate          : {rate_final} PDFs/min",
        title="[bold]Summary[/]",
        border_style="green",
    ))


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    asyncio.run(run(batch_size=batch))
