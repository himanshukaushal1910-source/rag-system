"""
sanity_check.py

Run this AFTER re-ingestion to verify all fixes took effect.

Usage: python sanity_check.py
       (runs inside asyncio.run so it uses the same AsyncQdrantClient as production)
"""

from __future__ import annotations

import asyncio

from config import get_settings
from retrieval.qdrant_client import QdrantClientSingleton, ensure_collection_exists
from qdrant_client.models import Filter, FieldCondition, MatchValue


async def main() -> None:
    settings = get_settings()
    COLLECTION = settings.qdrant_collection_name

    await ensure_collection_exists()
    client = await QdrantClientSingleton.get()

    print("\n" + "=" * 60)
    print("SANITY CHECK — RAG Pipeline Fixes")
    print("=" * 60)

    # ── Check 1: Table chunks exist ──────────────────────────────────
    print("\n[1] TABLE CHUNKS")
    results, _ = await client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="content_type", match=MatchValue(value="table"))]
        ),
        limit=3,
        with_payload=True,
    )
    if results:
        print(f"    Found {len(results)} table chunks (showing first 3)")
        for r in results:
            text_preview = r.payload.get("text", "")[:150].replace("\n", " | ")
            print(f"    -> {r.payload.get('filename')} p{r.payload.get('page_number')}: {text_preview}")
    else:
        print("    NO table chunks found")

    # ── Check 2: section_heading populated ───────────────────────────
    print("\n[2] SECTION HEADINGS ON TEXT CHUNKS")
    results, _ = await client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="content_type", match=MatchValue(value="text"))]
        ),
        limit=10,
        with_payload=True,
    )
    with_heading = [r for r in results if r.payload.get("section_heading", "").strip()]
    without_heading = [r for r in results if not r.payload.get("section_heading", "").strip()]
    print(f"    Chunks with heading:    {len(with_heading)}/10")
    print(f"    Chunks without heading: {len(without_heading)}/10")
    for r in with_heading[:3]:
        print(f"      -> '{r.payload.get('section_heading')}'")

    # ── Check 3: closing section chunks preserved ─────────────────────
    print("\n[3] CLOSING SECTION CHUNKS PRESERVED")
    conclusion_keywords = ["conclusion", "limitation", "discussion", "future"]
    results, _ = await client.scroll(collection_name=COLLECTION, limit=100, with_payload=True)
    found_closing = [
        r for r in results
        if any(kw in r.payload.get("section_heading", "").lower() for kw in conclusion_keywords)
    ]
    if found_closing:
        print(f"    Found {len(found_closing)} chunks from closing sections")
        for r in found_closing[:3]:
            print(f"      -> heading: '{r.payload.get('section_heading')}' | len: {len(r.payload.get('text',''))}")
    else:
        print("    No closing section chunks found")

    # ── Check 4: Total chunk count ────────────────────────────────────
    print("\n[4] TOTAL CHUNK COUNT")
    info = await client.get_collection(COLLECTION)
    print(f"    Total points in collection: {info.points_count}")

    # ── Check 5: Image chunks ─────────────────────────────────────────
    print("\n[5] IMAGE CHUNKS")
    results, _ = await client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="content_type", match=MatchValue(value="image"))]
        ),
        limit=3,
        with_payload=True,
    )
    if results:
        print(f"    Found {len(results)} image chunks")
        for r in results:
            text = r.payload.get("text", "")
            has_b64 = bool(r.payload.get("image_b64"))
            print(f"      -> alt_text: '{text[:80]}' | has_image_b64: {has_b64}")
    else:
        print("    No image chunks found (OK if PDFs have no embedded images)")

    await QdrantClientSingleton.close()

    print("\n" + "=" * 60)
    print("Check complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
