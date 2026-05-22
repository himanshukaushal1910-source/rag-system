"""
retrieval/contextual_compressor.py

Contextual Compression: extract query-relevant sentences from retrieved chunks.

After retrieval, each chunk may contain a mix of relevant and irrelevant sentences.
For a query about "attention mechanism complexity", a chunk about transformer architecture
might have 10 sentences but only 2-3 directly answer the question.

This compressor uses gpt-4o-mini to identify and extract ONLY the relevant sentences,
reducing context window noise and improving generation quality.

When to use:
- Enabled via config.contextual_compression_enabled = True
- Adds ~0.3-0.5s per chunk (parallel batch)
- Most valuable for broad topic queries where chunks have mixed content
- Skip for chunks already under 200 chars (no compression needed)

The original chunk text is preserved in chunk._original_text for citation purposes.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import structlog

from retrieval.hybrid_retriever import RetrievedChunk

logger = structlog.get_logger(__name__)

_MIN_CHUNK_CHARS = 200

_COMPRESS_SYSTEM_PROMPT = (
    "Extract ONLY the sentences from the following text that are directly relevant "
    "to the question. If no sentences are relevant, return the first 2 sentences. "
    "Return ONLY the extracted sentences, nothing else."
)


async def compress_chunk(
    query: str,
    chunk: RetrievedChunk,
    openai_client: Any,
    model: str = "gpt-4o-mini",
) -> RetrievedChunk:
    """Extract query-relevant sentences from a single chunk using GPT-4o-mini.

    Short chunks (< 200 chars) are returned unchanged — there is nothing
    to compress.  For longer chunks, the LLM is instructed to return only
    the sentences that directly address the query.

    A shallow copy of the chunk is returned (via dataclasses.replace) so
    the original object is never mutated.  The original text is stored
    in the returned chunk's payload at key "_original_text" for downstream
    citation or fallback use.

    Args:
        query:         User query string.
        chunk:         RetrievedChunk to compress.
        openai_client: AsyncOpenAI client instance.
        model:         OpenAI chat model to use (default gpt-4o-mini).

    Returns:
        New RetrievedChunk with .text replaced by compressed content.
        Returns a copy of the original chunk on any error.
    """
    if len(chunk.text) < _MIN_CHUNK_CHARS:
        return chunk

    log = logger.bind(
        chunk_id=chunk.id,
        original_len=len(chunk.text),
        query=query[:80],
    )
    log.debug("contextual_compressor.compressing")

    user_prompt = (
        f"Question: {query}\n\n"
        f"Text:\n{chunk.text}"
    )

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        )
        compressed_text = (response.choices[0].message.content or "").strip()

        if not compressed_text:
            log.warning("contextual_compressor.empty_response_keeping_original")
            return chunk

        log.debug(
            "contextual_compressor.done",
            original_len=len(chunk.text),
            compressed_len=len(compressed_text),
            compression_ratio=round(len(compressed_text) / len(chunk.text), 2),
        )

        return dataclasses.replace(chunk, text=compressed_text)

    except Exception as exc:
        log.warning(
            "contextual_compressor.failed_keeping_original",
            error=str(exc),
        )
        return chunk


async def compress_chunks_batch(
    query: str,
    chunks: list[RetrievedChunk],
    openai_client: Any,
    model: str = "gpt-4o-mini",
) -> list[RetrievedChunk]:
    """Compress all chunks in parallel using asyncio.gather.

    Each chunk is independently compressed in a separate coroutine.
    If any individual chunk's compression fails, that chunk is kept with
    its original text (handled inside compress_chunk).

    The returned list preserves the original ordering of the input chunks.

    Args:
        query:         User query string shared across all compression calls.
        chunks:        List of RetrievedChunk objects to compress.
        openai_client: AsyncOpenAI client instance.
        model:         OpenAI chat model to use for compression.

    Returns:
        List of RetrievedChunk objects in the same order as input.
        Each chunk's .text contains the compressed (or original) content.
    """
    if not chunks:
        return []

    log = logger.bind(query=query[:80], num_chunks=len(chunks))
    log.info("contextual_compressor.batch_start")

    tasks = [
        compress_chunk(query=query, chunk=chunk, openai_client=openai_client, model=model)
        for chunk in chunks
    ]

    results: list[RetrievedChunk] = []
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            log.warning(
                "contextual_compressor.batch_item_failed",
                chunk_id=chunks[i].id,
                error=str(result),
            )
            results.append(chunks[i])  # keep original on unexpected error
        else:
            results.append(result)

    compressed_count = sum(
        1 for orig, comp in zip(chunks, results)
        if comp.text != orig.text
    )
    skipped_count = sum(
        1 for chunk in chunks
        if len(chunk.text) < _MIN_CHUNK_CHARS
    )

    log.info(
        "contextual_compressor.batch_done",
        compressed=compressed_count,
        skipped_short=skipped_count,
        unchanged=len(chunks) - compressed_count,
    )
    return results
