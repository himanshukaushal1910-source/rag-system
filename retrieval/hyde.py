"""
retrieval/hyde.py

Feature B1 — Hypothetical Document Embeddings (HyDE).

Instead of embedding the raw query for dense retrieval, HyDE:
1. Uses GPT-4o-mini to generate a ~200 word hypothetical answer
2. Embeds that hypothetical answer
3. Uses that embedding for dense retrieval

Why this works:
  The hypothetical answer lives in the same embedding space as real
  document chunks (both are answer-like text). Raw queries are often
  short and use question phrasing that doesn't match chunk phrasing.
  HyDE dramatically improves recall for complex/abstract questions.

Reference: Gao et al. "Precise Zero-Shot Dense Retrieval without
  Relevance Labels" (2022).
"""

from __future__ import annotations

import structlog
from langchain_openai import ChatOpenAI

from agent.prompts import hyde_prompt
from config import get_settings

logger = structlog.get_logger(__name__)


async def generate_hypothetical_document(query: str) -> str:
    """Generate a hypothetical answer passage for HyDE retrieval.

    Uses a cheap fast model (gpt-4o-mini by default) to generate a
    ~200 word passage that would answer the query if it existed in
    a real research paper. This passage is then embedded and used
    for dense vector search instead of the raw query.

    Args:
        query: Original user query string.

    Returns:
        Hypothetical document passage as a string. Falls back to the
        original query if generation fails, so retrieval still works.
    """
    settings = get_settings()

    if not settings.hyde_enabled:
        return query

    log = logger.bind(query=query[:80])
    log.debug("hyde.generating")

    try:
        llm = ChatOpenAI(
            model=settings.hyde_model,
            temperature=0.7,   # some creativity helps diversity
            openai_api_key=settings.openai_api_key,
            max_tokens=300,
        )
        chain = hyde_prompt | llm
        response = await chain.ainvoke({"query": query})
        hypothetical = response.content.strip()

        if not hypothetical or len(hypothetical) < 20:
            log.warning("hyde.empty_response")
            return query

        log.debug("hyde.done", length=len(hypothetical))
        return hypothetical

    except Exception as exc:
        log.warning("hyde.failed_falling_back", error=str(exc))
        return query
