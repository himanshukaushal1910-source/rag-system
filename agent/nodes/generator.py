"""
agent/nodes/generator.py

Multimodal generation node — GPT-4o with on-demand page rendering.

New in this version:
  - On-demand page rendering: when query asks about figures/charts/graphs,
    renders the relevant PDF pages as high-res images and passes them to
    GPT-4o vision. Works for ALL figure types including vector graphics.
  - Extracts image chunks and table chunks from reranked_chunks
  - Stores them in state as retrieved_images and retrieved_tables
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from agent.state import AgentState, Citation
from api.exceptions import GenerationError, PromptBuildError
from config import get_settings
from retrieval.hybrid_retriever import RetrievedChunk

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_openai_client: Any | None = None


def set_openai_client(client: Any) -> None:
    """Inject AsyncOpenAI client at lifespan startup."""
    global _openai_client
    _openai_client = client


def _get_openai_client() -> Any:
    if _openai_client is None:
        raise GenerationError(
            "OpenAI client not initialised — call set_openai_client() at lifespan startup.",
            detail="_openai_client is None",
        )
    return _openai_client


def _build_system_prompt(sections: list[str]) -> str:
    section_list = (
        ", ".join(f'"{s}"' for s in sections) if sections else "multiple sections"
    )
    return f"""You are a precise research assistant. You are given context chunks and page images from one or more PDF documents and must answer the user's question.

CRITICAL RULES:
1. Use ALL provided context chunks AND page images. Do not stop after the first relevant chunk.
2. The context spans these sections: {section_list}. Cover ALL of them when the question asks about multiple sections.
3. Every factual claim must be followed by its citation in the format [Doc: filename.pdf, Page: N].
4. For table data, reproduce exact numbers — do not paraphrase figures. Format tables as markdown tables.
5. For figures and charts: describe what you SEE in the provided page images — colors, trends, axes, values, patterns.
6. NEVER say "I cannot find" or "not mentioned in the context" if the information appears in ANY chunk or page image.
7. If information truly does not appear in any chunk or image, say: "The provided context does not contain information about [topic]."
8. For long-form questions, structure your answer with clear headings matching the sections in the context.
9. Do not hallucinate citations. Only cite filenames and page numbers from the provided context."""


def _build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    page_images: list[dict],
) -> list[dict]:
    """Build GPT-4o message list with text chunks, tables, and page images.

    Args:
        query:       User query string.
        chunks:      Reranked RetrievedChunk objects.
        page_images: List of rendered page image dicts from page_renderer.

    Returns:
        OpenAI chat message list.
    """
    sections: list[str] = list(
        dict.fromkeys(c.section_heading for c in chunks if c.section_heading)
    )
    system_prompt = _build_system_prompt(sections)
    context_parts: list[Any] = []

    # ── Text and table chunks ─────────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        citation_tag = f"[Doc: {chunk.filename}, Page: {chunk.page_number}]"
        heading_prefix = (
            f"[Section: {chunk.section_heading}]\n" if chunk.section_heading else ""
        )

        if chunk.content_type == "table":
            block_text = (
                f"{heading_prefix}"
                f"<table id='{i + 1}' citation='{citation_tag}'>\n"
                f"{chunk.text}\n"
                f"</table>"
            )
            context_parts.append({"type": "text", "text": block_text})

        elif chunk.content_type in ("image", "chart") and chunk.image_b64:
            # Embedded raster image (rare in academic PDFs)
            context_parts.append({
                "type": "text",
                "text": (
                    f"{heading_prefix}[Embedded Image {i + 1}] {citation_tag}\n"
                    f"{chunk.text}"
                ),
            })
            context_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{chunk.image_b64}",
                    "detail": "low",
                },
            })

        else:
            block_text = (
                f"{heading_prefix}"
                f"[Chunk {i + 1}] {citation_tag}\n"
                f"{chunk.text}"
            )
            context_parts.append({"type": "text", "text": block_text})

    # ── Rendered page images (vector figures, charts, diagrams) ───────────
    if page_images:
        context_parts.append({
            "type": "text",
            "text": "\n\n--- RENDERED PAGE IMAGES (contains figures/charts) ---",
        })
        for img in page_images:
            context_parts.append({
                "type": "text",
                "text": (
                    f"[Page Render: {img['filename']}, Page {img['page']}]\n"
                    f"Context: {img.get('caption', '')}"
                ),
            })
            context_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img['image_b64']}",
                    "detail": "high",  # high detail for figures
                },
            })

    # ── Question ──────────────────────────────────────────────────────────
    context_parts.append({
        "type": "text",
        "text": (
            f"\n\n---\n"
            f"QUESTION: {query}\n\n"
            f"Answer using ALL context chunks and page images above. "
            f"For figures and charts, describe what you see in the images. "
            f"Cite every factual claim as [Doc: filename.pdf, Page: N]. "
            f"Format any tabular data as markdown tables."
        ),
    })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_parts},
    ]


def _extract_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """Extract and validate citations from generated answer."""
    pattern = re.compile(r"\[Doc:\s*(.+?),\s*Page:\s*(\d+)\]")
    chunk_lookup: dict[tuple[str, int], RetrievedChunk] = {}
    for chunk in chunks:
        chunk_lookup[(chunk.filename, chunk.page_number)] = chunk

    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for match in pattern.finditer(answer):
        filename = match.group(1).strip()
        page = int(match.group(2))
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        chunk = chunk_lookup.get(key)
        citations.append(
            Citation(
                filename=filename,
                page=page,
                chunk_id=chunk.id if chunk else "unknown",
            )
        )
    return citations


def _extract_tables_from_answer(answer: str) -> list[dict]:
    """Extract markdown tables from generated answer."""
    tables = []
    lines = answer.split("\n")
    current_table: list[str] = []

    for line in lines:
        if line.strip().startswith("|"):
            current_table.append(line)
        else:
            if len(current_table) >= 2:
                tables.append({
                    "markdown": "\n".join(current_table),
                    "caption": "",
                })
            current_table = []

    if len(current_table) >= 2:
        tables.append({
            "markdown": "\n".join(current_table),
            "caption": "",
        })

    return tables


async def generator_node(state: AgentState) -> AgentState:
    """Generate a grounded answer with on-demand figure rendering.

    Pipeline:
    1. Get reranked chunks from state
    2. Check if query asks about figures/charts
    3. If yes: render relevant PDF pages as high-res images
    4. Build GPT-4o message with chunks + rendered pages
    5. GPT-4o vision reads both text and images
    6. Extract citations, images, tables from response

    Args:
        state: Current agent state with reranked_chunks and original_query.

    Returns:
        Updated state with generated_answer, citations,
        retrieved_images, and retrieved_tables.
    """
    settings = get_settings()
    chunks: list[RetrievedChunk] = (
        state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    )
    query = state["original_query"]
    log = logger.bind(node="generator", chunks=len(chunks), query=query[:80])

    if not chunks:
        log.warning("generator_node.no_chunks")
        return {
            **state,
            "generated_answer": (
                "I cannot find sufficient information in the provided documents "
                "to answer this question."
            ),
            "citations": [],
            "retrieved_images": [],
            "retrieved_tables": [],
        }

    # ── On-demand page rendering for figure queries ───────────────────────
    page_images: list[dict] = []
    try:
        from retrieval.page_renderer import get_figure_pages
        page_images = get_figure_pages(
            query=query,
            chunks=chunks,
            max_pages=3,  # render up to 3 pages per query
        )
        if page_images:
            log.info(
                "generator_node.pages_rendered",
                count=len(page_images),
                pages=[(p["filename"], p["page"]) for p in page_images],
            )
    except Exception as exc:
        log.warning("generator_node.page_render_failed", error=str(exc))

    # ── Build messages ────────────────────────────────────────────────────
    log.info("generator_node.building_messages", page_images=len(page_images))

    try:
        messages = _build_messages(query, chunks, page_images)
    except Exception as exc:
        raise PromptBuildError(
            "Failed to build context messages for generator",
            detail=str(exc),
        ) from exc

    client = _get_openai_client()
    log.info(
        "generator_node.calling_llm",
        model=settings.llm_model,
        chunks=len(chunks),
        page_images=len(page_images),
    )

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise GenerationError(
            "GPT-4o generation failed",
            detail=str(exc),
        ) from exc

    citations = _extract_citations(answer, chunks)

    # Combine embedded image chunks + rendered page images for API response
    embedded_images = [
        {
            "filename": c.filename,
            "page": c.page_number,
            "caption": c.text if c.text != "[image content]" else "",
            "image_b64": c.image_b64,
        }
        for c in chunks
        if c.content_type in ("image", "chart") and c.image_b64
    ]
    all_images = embedded_images + page_images

    retrieved_tables = _extract_tables_from_answer(answer)

    log.info(
        "generator_node.done",
        answer_length=len(answer),
        citations=len(citations),
        images=len(all_images),
        tables=len(retrieved_tables),
    )

    return {
        **state,
        "generated_answer": answer,
        "citations": citations,
        "retrieved_images": all_images,
        "retrieved_tables": retrieved_tables,
    }
