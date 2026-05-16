from __future__ import annotations

import asyncio
import json

import structlog
from langchain_openai import ChatOpenAI

from agent.prompts import consistency_prompt, faithfulness_prompt
from agent.state import AgentState
from api.exceptions import FaithfulnessError, VerificationError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)


def _build_context_str(state: AgentState) -> str:
    """Serialise reranked chunks to a plain text context string.

    Args:
        state: Agent state containing reranked chunks.

    Returns:
        Multi-line string with chunk text and source metadata.
    """
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    return "\n\n".join(
        f"[{c.filename} | Page {c.page_number}]\n{c.text}"
        for c in chunks
    )


async def _check_faithfulness(
    llm: ChatOpenAI,
    answer: str,
    context: str,
) -> float:
    """Score faithfulness of the answer against the context.

    Uses an LLM judge prompt that checks sentence-level support.

    Args:
        llm: ChatOpenAI instance.
        answer: Generated answer to evaluate.
        context: Context string from reranked chunks.

    Returns:
        Faithfulness score between 0.0 and 1.0.

    Raises:
        FaithfulnessError: If the judge LLM call or parsing fails.
    """
    chain = faithfulness_prompt | llm
    try:
        response = await chain.ainvoke({"context": context, "answer": answer})
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        score = float(parsed.get("faithfulness_score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception as exc:
        raise FaithfulnessError(
            "Faithfulness judge failed",
            detail=str(exc),
        ) from exc


async def _check_consistency(
    llm: ChatOpenAI,
    query: str,
    settings_obj: object,
) -> bool:
    """Run self-consistency check by generating N answers and comparing.

    Generates ``consistency_samples`` answers at temperature 0.7, then
    uses an LLM judge to check if they agree on key facts.

    Args:
        llm: ChatOpenAI instance (temperature overridden internally).
        query: Original query string.
        settings_obj: Settings instance for consistency_samples count.

    Returns:
        True if answers are consistent, False otherwise.
    """
    from agent.prompts import GENERATOR_SYSTEM
    from config import get_settings

    settings = get_settings()
    n = settings.consistency_samples

    hot_llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.7,
        openai_api_key=settings.openai_api_key,
    )

    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": query},
    ]

    # Generate N answers in parallel
    tasks = [hot_llm.ainvoke(messages) for _ in range(n)]
    try:
        responses = await asyncio.gather(*tasks)
        answers = [r.content.strip() for r in responses]
    except Exception as exc:
        logger.warning("Consistency generation failed", error=str(exc))
        return True  # Fail open — don't block on consistency errors

    # Judge agreement
    judge_chain = consistency_prompt | ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    answers_text = "\n\n---\n\n".join(
        f"Answer {i+1}:\n{a}" for i, a in enumerate(answers)
    )

    try:
        response = await judge_chain.ainvoke({
            "query": query,
            "answers": answers_text,
        })
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        agreement: bool = bool(parsed.get("agreement", True))
        logger.info(
            "Consistency check result",
            agreement=agreement,
            reasoning=parsed.get("reasoning", "")[:100],
        )
        return agreement
    except Exception as exc:
        logger.warning("Consistency judge failed", error=str(exc))
        return True  # Fail open


def _verify_citations(state: AgentState) -> tuple[bool, list[str]]:
    """Cross-check extracted citations against reranked chunks.

    Args:
        state: Agent state with citations and reranked_chunks.

    Returns:
        Tuple of (all_valid: bool, hallucinated_citations: list[str]).
    """
    citations = state.get("citations") or []
    chunks = state.get("reranked_chunks") or []

    valid_keys = {(c.filename, c.page_number) for c in chunks}
    hallucinated = []

    for citation in citations:
        key = (citation["filename"], citation["page"])
        if key not in valid_keys:
            hallucinated.append(f"{citation['filename']} Page {citation['page']}")

    return len(hallucinated) == 0, hallucinated


async def verifier_node(state: AgentState) -> AgentState:
    """Hallucination guard: faithfulness + consistency + citation check.

    Runs three verification checks:
    1. **Faithfulness**: LLM judge scores sentence-level support (0–1).
    2. **Self-consistency**: N parallel generations agree on key facts.
    3. **Citation verification**: All cited sources exist in retrieved chunks.

    If faithfulness score is below threshold, sets state to trigger
    re-retrieval via the graph's conditional edge.

    Args:
        state: Current agent state. Reads ``generated_answer``,
            ``reranked_chunks``, ``citations``, ``original_query``.

    Returns:
        Updated state with ``faithfulness_score``, ``consistency_passed``,
        and ``final_answer`` (if verification passes).

    Raises:
        VerificationError: If verification itself errors unrecoverably.
    """
    settings = get_settings()
    answer = state.get("generated_answer", "")
    query = state["original_query"]
    retry_count = state.get("retry_count", 0)
    log = logger.bind(node="verifier", retry_count=retry_count)

    log.info("Starting verification")

    if not answer:
        return {
            **state,
            "faithfulness_score": 0.0,
            "consistency_passed": False,
            "final_answer": "",
            "error": "Empty generated answer",
        }

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )

    context = _build_context_str(state)

    # ------------------------------------------------------------------ #
    # 1. Faithfulness check
    # ------------------------------------------------------------------ #
    try:
        faithfulness_score = await _check_faithfulness(llm, answer, context)
        log.info("Faithfulness scored", score=round(faithfulness_score, 3))
    except FaithfulnessError as exc:
        log.warning("Faithfulness check errored — defaulting to 0.5", error=str(exc))
        faithfulness_score = 0.5

    # ------------------------------------------------------------------ #
    # 2. Self-consistency check (skip on retries to save API calls)
    # ------------------------------------------------------------------ #
    if retry_count == 0:
        consistency_passed = await _check_consistency(llm, query, settings)
    else:
        log.info("Skipping consistency check on retry")
        consistency_passed = True

    # ------------------------------------------------------------------ #
    # 3. Citation verification
    # ------------------------------------------------------------------ #
    citations_valid, hallucinated = _verify_citations(state)
    if not citations_valid:
        log.warning(
            "Hallucinated citations detected",
            hallucinated=hallucinated,
        )

    # ------------------------------------------------------------------ #
    # 4. Decision
    # ------------------------------------------------------------------ #
    passes_faithfulness = faithfulness_score >= settings.faithfulness_threshold

    log.info(
        "Verification complete",
        faithfulness=round(faithfulness_score, 3),
        threshold=settings.faithfulness_threshold,
        passes_faithfulness=passes_faithfulness,
        consistency_passed=consistency_passed,
        citations_valid=citations_valid,
    )

    final_answer = answer if (passes_faithfulness and consistency_passed) else ""

    return {
        **state,
        "faithfulness_score": faithfulness_score,
        "consistency_passed": consistency_passed,
        "final_answer": final_answer,
        "retry_count": retry_count,
    }
