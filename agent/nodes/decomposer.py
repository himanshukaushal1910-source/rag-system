from __future__ import annotations

import json

import structlog
from langchain_openai import ChatOpenAI

from agent.prompts import decomposer_prompt
from agent.state import AgentState
from api.exceptions import GenerationError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)


async def decomposer_node(state: AgentState) -> AgentState:
    """Decompose the original query into focused sub-queries.

    Uses GPT-4o with structured output to produce a JSON array of
    sub-questions. Falls back to ``[original_query]`` on any failure so
    the pipeline continues even if decomposition errors.

    Args:
        state: Current agent state. Reads ``original_query``.

    Returns:
        Updated state with ``sub_queries`` populated.
    """
    settings = get_settings()
    query = state["original_query"]
    log = logger.bind(node="decomposer", query=query[:80])
    log.info("Decomposing query")

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    chain = decomposer_prompt | llm

    try:
        response = await chain.ainvoke({"query": query})
        raw = response.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        sub_queries: list[str] = json.loads(raw)

        if not isinstance(sub_queries, list) or not sub_queries:
            raise ValueError(f"Expected non-empty list, got: {raw}")

        # Sanitise — ensure all items are strings
        sub_queries = [str(q).strip() for q in sub_queries if str(q).strip()]

        log.info("Query decomposed", sub_queries=sub_queries)
        return {**state, "sub_queries": sub_queries}

    except Exception as exc:
        log.warning(
            "Decomposition failed — falling back to original query",
            error=str(exc),
        )
        # Graceful fallback: treat the original query as the only sub-query.
        return {**state, "sub_queries": [query]}
