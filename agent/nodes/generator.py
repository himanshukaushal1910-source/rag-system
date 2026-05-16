from __future__ import annotations

import re

import structlog
from langchain_openai import ChatOpenAI

from agent.state import AgentState, Citation
from api.exceptions import GenerationError, PromptBuildError
from config import get_settings
from retrieval.hybrid_retriever import RetrievedChunk

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Regex to extract citations from generated text.
# Matches: [Doc: filename.pdf, Page: 3]
_CITATION_RE = re.compile(
    r"\[Doc:\s*(?P<filename>[^,\]]+),\s*Page:\s*(?P<page>\d+)\]"
)


def _build_context_messages(chunks: list[RetrievedChunk]) -> list[dict]:
    """Build the context portion of the GPT-4o message payload.

    Text chunks are passed as text blocks. Image/chart chunks include the
    base64 image inline so GPT-4o vision can process them.

    Args:
        chunks: Re-ranked chunks to include as context.

    Returns:
        List of OpenAI message content dicts (text + image_url blocks).
    """
    parts: list[dict] = []

    for i, chunk in enumerate(chunks, 1):
        header = (
            f"[Chunk {i} | {chunk.filename} | Page {chunk.page_number} "
            f"| Type: {chunk.content_type}]\n"
        )

        if chunk.content_type in ("image", "chart") and chunk.image_b64:
            # Text header for the image
            parts.append({"type": "text", "text": header})
            # Base64 image for GPT-4o vision
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{chunk.image_b64}",
                    "detail": "auto",
                },
            })
        else:
            parts.append({
                "type": "text",
                "text": f"{header}{chunk.text}\n",
            })

    return parts


def _extract_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> list[Citation]:
    """Extract and verify citations from a generated answer.

    Parses ``[Doc: filename.pdf, Page: N]`` patterns and cross-references
    them against the provided chunks. Unverified citations are still
    included but can be flagged by the verifier node.

    Args:
        answer: Generated answer text.
        chunks: Re-ranked chunks used for generation.

    Returns:
        List of :class:`Citation` dicts.
    """
    chunk_index: dict[tuple[str, int], str] = {
        (c.filename, c.page_number): c.chunk_id for c in chunks
    }

    citations: list[Citation] = []
    seen: set[tuple[str, int]] = set()

    for match in _CITATION_RE.finditer(answer):
        filename = match.group("filename").strip()
        page = int(match.group("page"))
        key = (filename, page)

        if key in seen:
            continue
        seen.add(key)

        chunk_id = chunk_index.get(key, "unknown")
        citations.append(Citation(filename=filename, page=page, chunk_id=chunk_id))

    return citations


async def generator_node(state: AgentState) -> AgentState:
    """Generate a grounded, cited answer from re-ranked chunks.

    Builds a multimodal GPT-4o prompt combining text and image chunks,
    enforces strict citation rules, then extracts structured citations
    from the response.

    Args:
        state: Current agent state. Reads ``reranked_chunks`` and
            ``original_query``.

    Returns:
        Updated state with ``generated_answer`` and ``citations``.

    Raises:
        GenerationError: If the OpenAI API call fails.
        PromptBuildError: If context message construction fails.
    """
    settings = get_settings()
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    query = state["original_query"]
    log = logger.bind(node="generator", chunks=len(chunks), query=query[:80])

    if not chunks:
        log.warning("No chunks available for generation")
        return {
            **state,
            "generated_answer": "I cannot find sufficient information in the provided documents to answer this question.",
            "citations": [],
        }

    log.info("Generating answer")

    # ------------------------------------------------------------------ #
    # Build multimodal context message content
    # ------------------------------------------------------------------ #
    try:
        context_parts = _build_context_messages(chunks)
    except Exception as exc:
        raise PromptBuildError(
            "Failed to build context message",
            detail=str(exc),
        ) from exc

    # Build the full message list for GPT-4o
    from agent.prompts import GENERATOR_SYSTEM

    # Context as a plain text string for text-only fallback
    context_text = "\n\n".join(
        f"[{c.filename} | Page {c.page_number}]\n{c.text}" for c in chunks
        if c.content_type not in ("image", "chart")
    )

    messages: list[dict] = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {
            "role": "user",
            "content": context_parts + [
                {
                    "type": "text",
                    "text": (
                        f"\nQuestion: {query}\n\n"
                        "Provide a comprehensive answer with inline citations "
                        "[Doc: filename.pdf, Page: N] for every factual claim."
                    ),
                }
            ],
        },
    ]

    # ------------------------------------------------------------------ #
    # Call GPT-4o
    # ------------------------------------------------------------------ #
    try:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key,
        )
        response = await llm.ainvoke(messages)
        answer = response.content.strip()
    except Exception as exc:
        raise GenerationError(
            "GPT-4o generation failed",
            detail=str(exc),
        ) from exc

    citations = _extract_citations(answer, chunks)

    log.info(
        "Generation complete",
        answer_length=len(answer),
        citations=len(citations),
    )

    return {
        **state,
        "generated_answer": answer,
        "citations": citations,
    }
