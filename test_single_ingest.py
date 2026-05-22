import asyncio
from pathlib import Path
from config import get_settings
from ingestion.chunker import build_chunker
from ingestion.embedder import AsyncEmbedder
from ingestion.sparse_encoder import SparseEncoder
from ingestion.ingestor import ingest_pdf
from retrieval.qdrant_client import ensure_collection_exists

async def test():
    s = get_settings()
    print("Ensuring collection...")
    await ensure_collection_exists()
    print("Building chunker...")
    chunker = build_chunker(s)
    print("Chunker ready")
    embedder = AsyncEmbedder()
    sparse = SparseEncoder()

    pdf = list(Path(r"D:\rag_system\data\pdfs\papers").glob("*.pdf"))[0]
    print("Ingesting:", pdf.name)

    result = await ingest_pdf(
        pdf,
        chunker=chunker,
        embedder=embedder,
        sparse_encoder=sparse,
    )
    print("Result:", result)

asyncio.run(test())
