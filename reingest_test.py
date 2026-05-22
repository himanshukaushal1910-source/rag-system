"""
reingest_test.py

Wipes the Qdrant collection and re-ingests ONLY the 10 test PDFs
from data/pdfs/test files/ to verify all sanity check fixes applied
correctly before running full ingestion on 1000+ papers.

Usage:
    python reingest_test.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from rich.console import Console

from config import get_settings
from ingestion.chunker import build_chunker
from ingestion.embedder import AsyncEmbedder
from ingestion.ingestor import ingest_pdf
from ingestion.sparse_encoder import SparseEncoder
from retrieval.qdrant_client import QdrantClientSingleton, ensure_collection_exists

logger = structlog.get_logger(__name__)
console = Console()

TEST_DIR = Path(r"D:\rag_system\data\pdfs\test files")


async def run() -> None:
    """Wipe collection, re-ingest test PDFs, print per-file summary."""
    settings = get_settings()

    # ── Step 1: Wipe existing collection ──────────────────────────────────────
    console.print("\n[bold yellow]Step 1: Wiping existing Qdrant collection...[/]")
    client = await QdrantClientSingleton.get()

    collections = await client.get_collections()
    existing = [c.name for c in collections.collections]

    if settings.qdrant_collection_name in existing:
        await client.delete_collection(settings.qdrant_collection_name)
        console.print(f"[green]✓[/] Deleted collection '{settings.qdrant_collection_name}'")
    else:
        console.print(f"[yellow]Collection '{settings.qdrant_collection_name}' did not exist — skipping delete[/]")

    # ── Step 2: Recreate collection ────────────────────────────────────────────
    console.print("\n[bold yellow]Step 2: Recreating collection...[/]")
    await ensure_collection_exists()
    console.print("[green]✓[/] Collection recreated")

    # ── Step 3: Load components once ──────────────────────────────────────────
    console.print("\n[bold yellow]Step 3: Loading components...[/]")
    chunker = build_chunker(settings)
    embedder = AsyncEmbedder()
    sparse_encoder = SparseEncoder()
    console.print("[green]✓[/] Components ready")

    # ── Step 4: Ingest test PDFs ───────────────────────────────────────────────
    pdfs = sorted(TEST_DIR.glob("*.pdf"))
    console.print(f"\n[bold yellow]Step 4: Ingesting {len(pdfs)} test PDFs from:[/]")
    console.print(f"  {TEST_DIR}\n")

    total_chunks = 0
    failed = 0

    for pdf_path in pdfs:
        try:
            result = await ingest_pdf(
                pdf_path,
                chunker=chunker,
                embedder=embedder,
                sparse_encoder=sparse_encoder,
            )
            chunks = result["chunks_ingested"]
            pages = result["pages_ingested"]
            total_chunks += chunks
            console.print(
                f"  [green]✓[/] {pdf_path.name} — "
                f"{pages} pages, [cyan]{chunks:,}[/] chunks"
            )
        except Exception as exc:
            failed += 1
            console.print(f"  [red]✗[/] {pdf_path.name} — {exc}")

    # ── Summary ────────────────────────────────────────────────────────────────
    console.print(f"\n[bold]Done.[/] {len(pdfs) - failed}/{len(pdfs)} PDFs succeeded")
    console.print(f"Total chunks ingested: [cyan]{total_chunks:,}[/]")

    if failed:
        console.print(f"[red]{failed} PDFs failed — check logs above[/]")
    else:
        console.print("\n[bold green]All test PDFs ingested. Now run:[/]")
        console.print("  [cyan]python sanity_check.py[/]")


if __name__ == "__main__":
    asyncio.run(run())
