"""
agent/nodes/generator.py

Multimodal generation node — GPT-4o with on-demand page rendering.

Improvements in this version:
  - Numbered inline citations [1], [2] with a References section
  - Query-type-aware system prompt (factual/analytical/visual/table/code)
  - Better structured output with headings for analytical queries
  - Math formula detection and preservation
  - Code block detection and preservation
  - Multi-document synthesis when chunks from >2 papers
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


# ── Query-type system prompts ─────────────────────────────────────────────────

_BASE_RULES = """CRITICAL RULES:
1. Use ALL provided context chunks AND page images. The answer may span multiple chunks.
2. Number every citation as [N] where N matches the chunk number in the context. Collect all unique sources in a "References" section at the END of your answer.
3. NEVER say "I cannot find" or "not mentioned" if the information appears in ANY chunk or page image.
4. If information truly is absent: "The provided context does not contain information about [topic]."
5. Never hallucinate numbers, names, dates, or statistics not present in the context.
6. Do not repeat citations already listed — just reuse the same [N]."""

_FACTUAL_SYSTEM = f"""You are a precise research assistant answering factual questions from PDF documents.
{_BASE_RULES}
For factual questions: give the direct answer first, then supporting context, then cite [N].
Keep answers concise — one clear paragraph unless multiple facts are needed."""

_ANALYTICAL_SYSTEM = f"""You are a precise research assistant answering analytical questions from PDF documents.
{_BASE_RULES}
For analytical questions:
- Use clear headings matching the sections in the context
- Explain mechanisms step by step
- Cover ALL relevant sections — do not stop at the first chunk
- Format as structured prose with headers for multi-part answers"""

_COMPARATIVE_SYSTEM = f"""You are a precise research assistant synthesising findings across multiple PDF documents.
{_BASE_RULES}
For comparative questions:
1. **Overview** — what all sources agree on
2. **Per-source findings** — key points from each document
3. **Synthesis** — how findings relate and answer the question
If sources disagree, note the disagreement explicitly."""

_VISUAL_SYSTEM = f"""You are a precise research assistant answering questions about figures and charts from PDF documents.
{_BASE_RULES}
For visual questions:
- Describe what you SEE in the provided page images — axes, values, colors, trends, patterns
- Reference specific visual elements (e.g., "The blue curve shows...", "The y-axis represents...")
- If a chart compares methods, describe each method's performance explicitly
- Include figure numbers and paper names in your description"""

_TABLE_SYSTEM = f"""You are a precise research assistant answering questions about tables and data from PDF documents.
{_BASE_RULES}
For table questions:
- Reproduce exact numbers from the table — never paraphrase or round unless explicitly asked
- Format table data as a markdown table
- Include column headers and row labels exactly as they appear
- If comparing multiple tables, note any differences in metrics or scales"""

_CODE_SYSTEM = f"""You are a precise research assistant answering questions about algorithms and code from PDF documents.
{_BASE_RULES}
For code/algorithm questions:
- Reproduce pseudocode or algorithm steps exactly as presented
- Wrap code in fenced code blocks: ```python ... ``` or ```algorithm ... ```
- Explain each step in plain language after showing the code
- Note time/space complexity if mentioned in the context"""

_SYSTEM_BY_TYPE = {
    "factual": _FACTUAL_SYSTEM,
    "analytical": _ANALYTICAL_SYSTEM,
    "comparative": _COMPARATIVE_SYSTEM,
    "visual": _VISUAL_SYSTEM,
    "table": _TABLE_SYSTEM,
    "code": _CODE_SYSTEM,
}


def _get_system_prompt(query_type: str, sections: list[str], num_docs: int) -> str:
    base = _SYSTEM_BY_TYPE.get(query_type, _ANALYTICAL_SYSTEM)
    # Append multi-doc instruction when chunks span many papers
    if num_docs > 2 and query_type not in ("visual", "table", "code"):
        base += (
            f"\n\nContext spans {num_docs} different papers: "
            f"synthesise their findings — don't just list each paper's content separately."
        )
    if sections:
        section_list = ", ".join(f'"{s}"' for s in sections[:8])
        base += f"\n\nSections in context: {section_list}"
    return base


def _build_messages(
    query: str,
    chunks: list[RetrievedChunk],
    page_images: list[dict],
    query_type: str = "analytical",
) -> list[dict]:
    """Build GPT-4o message list with numbered chunks, tables, and page images."""
    sections: list[str] = list(
        dict.fromkeys(c.section_heading for c in chunks if c.section_heading)
    )
    unique_docs = len({c.filename for c in chunks})
    system_prompt = _get_system_prompt(query_type, sections, unique_docs)

    context_parts: list[Any] = []

    # ── Numbered context chunks ───────────────────────────────────────────────
    for i, chunk in enumerate(chunks):
        citation_tag = f"[{i + 1}] {chunk.filename}, p.{chunk.page_number}"
        heading_prefix = (
            f"[Section: {chunk.section_heading}]\n" if chunk.section_heading else ""
        )

        if chunk.content_type == "table":
            block_text = (
                f"{heading_prefix}"
                f"<table id='{i + 1}' source='{citation_tag}'>\n"
                f"{chunk.text}\n"
                f"</table>"
            )
            context_parts.append({"type": "text", "text": block_text})

        elif chunk.content_type in ("image", "chart") and chunk.image_b64:
            context_parts.append({
                "type": "text",
                "text": (
                    f"{heading_prefix}[Image {i + 1}] Source: {citation_tag}\n"
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
                f"[{i + 1}] Source: {citation_tag}\n"
                f"{chunk.text}"
            )
            context_parts.append({"type": "text", "text": block_text})

    # ── Rendered page images ──────────────────────────────────────────────────
    if page_images:
        context_parts.append({
            "type": "text",
            "text": "\n\n--- RENDERED PAGE IMAGES (figures/charts from PDF) ---",
        })
        for img in page_images:
            context_parts.append({
                "type": "text",
                "text": (
                    f"[Page Image] {img['filename']}, Page {img['page']}\n"
                    f"Caption context: {img.get('caption', '')}"
                ),
            })
            context_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img['image_b64']}",
                    "detail": "high",
                },
            })

    # ── Question with citation instructions ───────────────────────────────────
    context_parts.append({
        "type": "text",
        "text": (
            f"\n\n---\n"
            f"QUESTION: {query}\n\n"
            f"Instructions:\n"
            f"- Cite every factual claim with [N] where N is the chunk number above\n"
            f"- End your answer with a '## References' section listing each cited source as:\n"
            f"  [N] filename.pdf, p.PAGE\n"
            f"- For tables: output as markdown tables\n"
            f"- For code/algorithms: output as fenced code blocks\n"
            f"- For math formulas: use LaTeX notation: $formula$ for inline, $$formula$$ for display"
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
    """Extract numbered citations [N] from the answer and map to source chunks."""
    # Find all [N] references in the answer
    pattern = re.compile(r"\[(\d+)\]")
    found_nums = set(int(m.group(1)) for m in pattern.finditer(answer))

    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for num in sorted(found_nums):
        idx = num - 1  # chunks are 0-indexed
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            key = (chunk.filename, chunk.page_number)
            if key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        filename=chunk.filename,
                        page=chunk.page_number,
                        chunk_id=chunk.id,
                    )
                )

    return citations


def _extract_tables_from_answer(answer: str) -> list[dict]:
    """Extract markdown tables from generated answer with caption detection."""
    tables = []
    lines = answer.split("\n")
    current_table: list[str] = []
    prev_line = ""

    for line in lines:
        if line.strip().startswith("|"):
            current_table.append(line)
        else:
            if len(current_table) >= 2:
                # Check for a caption on the line immediately before the table
                caption = prev_line.strip() if prev_line.strip() and not prev_line.strip().startswith("|") else ""
                tables.append({
                    "markdown": "\n".join(current_table),
                    "caption": caption,
                })
            current_table = []
        prev_line = line

    if len(current_table) >= 2:
        tables.append({
            "markdown": "\n".join(current_table),
            "caption": "",
        })

    return tables


async def generator_node(state: AgentState) -> AgentState:
    """Generate a grounded answer with query-type-aware prompting and numbered citations.

    Pipeline:
    1. Get reranked chunks and query type from state
    2. On-demand page rendering for visual/figure queries
    3. Build query-type-specific message with numbered chunks
    4. GPT-4o generation with temperature=0.1
    5. Extract numbered citations, tables, images
    """
    settings = get_settings()
    chunks: list[RetrievedChunk] = (
        state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    )
    query = state["original_query"]
    query_type = state.get("query_type") or "analytical"
    log = logger.bind(
        node="generator",
        chunks=len(chunks),
        query=query[:80],
        query_type=query_type,
    )

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

    # ── On-demand page rendering for visual queries ───────────────────────────
    page_images: list[dict] = []
    # Expand rendering for visual queries; also render for analytical if figures mentioned
    should_render = query_type in ("visual",) or (
        query_type == "analytical" and re.search(r"\b(figure|fig|chart|plot|graph)\b", query, re.I)
    )
    if should_render:
        try:
            from retrieval.page_renderer import get_figure_pages
            max_pages = 5 if query_type == "visual" else 3
            page_images = get_figure_pages(
                query=query,
                chunks=chunks,
                max_pages=max_pages,
            )
            if page_images:
                log.info(
                    "generator_node.pages_rendered",
                    count=len(page_images),
                    pages=[(p["filename"], p["page"]) for p in page_images],
                )
        except Exception as exc:
            log.warning("generator_node.page_render_failed", error=str(exc))

    # ── Build messages ────────────────────────────────────────────────────────
    try:
        messages = _build_messages(query, chunks, page_images, query_type)
    except Exception as exc:
        raise PromptBuildError(
            "Failed to build context messages for generator",
            detail=str(exc),
        ) from exc

    client = _get_openai_client()

    # Increase token budget for analytical/comparative queries
    max_tokens = 4096 if query_type in ("analytical", "comparative") else 2048
    log.info(
        "generator_node.calling_llm",
        model=settings.llm_model,
        chunks=len(chunks),
        page_images=len(page_images),
        max_tokens=max_tokens,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=max_tokens,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise GenerationError(
            "GPT-4o generation failed",
            detail=str(exc),
        ) from exc

    citations = _extract_citations(answer, chunks)

    # Collect embedded image/chart chunks + rendered page images
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
