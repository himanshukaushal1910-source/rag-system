"""
agent/nodes/verifier.py

Hallucination guard node — faithfulness + consistency + citation check.

Feature D1: NLI faithfulness replaces LLM judge when use_nli_faithfulness=True.
Feature D2: Answer completeness check — verifies all sub-queries were addressed.
Feature D3: Per-claim confidence via NLI score per sentence.

Fixes applied:
  C-5   get_running_loop() instead of deprecated get_event_loop()
  M-4   Consistency check now includes retrieved context chunks
  M-9   Extended stop-word list (NLTK-style) for better completeness scoring
  M-12  ChatOpenAI clients instantiated once via module-level singletons
"""

from __future__ import annotations

import asyncio
import json
import re

import structlog
from langchain_openai import ChatOpenAI

from agent.prompts import consistency_prompt, faithfulness_prompt
from agent.state import AgentState
from api.exceptions import FaithfulnessError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# ── Module-level LLM singletons (M-12) ───────────────────────────────────────
_llm_zero: ChatOpenAI | None = None
_llm_hot: ChatOpenAI | None = None


def _get_llm_zero() -> ChatOpenAI:
    global _llm_zero
    if _llm_zero is None:
        s = get_settings()
        _llm_zero = ChatOpenAI(model=s.llm_model, temperature=0, openai_api_key=s.openai_api_key)
    return _llm_zero


def _get_llm_hot() -> ChatOpenAI:
    global _llm_hot
    if _llm_hot is None:
        s = get_settings()
        _llm_hot = ChatOpenAI(model=s.llm_model, temperature=0.7, openai_api_key=s.openai_api_key)
    return _llm_hot


# Extended stop-word set (M-9) — covers common English function words
_STOP_WORDS: frozenset[str] = frozenset({
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "shall", "can", "must", "ought",
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "nor",
    "but", "so", "yet", "both", "either", "neither", "not", "with", "from",
    "by", "as", "if", "then", "than", "that", "this", "these", "those",
    "its", "it", "they", "their", "them", "we", "our", "you", "your",
    "about", "into", "through", "during", "before", "after", "above", "below",
    "between", "each", "more", "most", "other", "some", "such", "no", "only",
    "same", "too", "very", "just", "also", "across", "along", "around",
    "describe", "explain", "discuss", "compare", "contrast", "outline",
})


def _build_context_str(state: AgentState) -> str:
    """Serialise reranked chunks to plain text context string."""
    chunks = state.get("reranked_chunks") or state.get("retrieved_chunks") or []
    return "\n\n".join(
        f"[{c.filename} | Page {c.page_number}]\n{c.text}"
        for c in chunks
    )


async def _check_faithfulness_llm(
    llm: ChatOpenAI,
    answer: str,
    context: str,
) -> float:
    """LLM-based faithfulness judge (fallback when NLI disabled)."""
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
        raise FaithfulnessError("Faithfulness LLM judge failed", detail=str(exc)) from exc


def _check_faithfulness_nli(answer: str, context: str, model_name: str) -> float:
    """D1: NLI-based faithfulness scoring (preferred — no API call)."""
    from retrieval.nli_scorer import score_faithfulness_nli
    return score_faithfulness_nli(answer, context, model_name)


def _check_completeness(answer: str, sub_queries: list[str]) -> float:
    """D2: Check what fraction of sub-queries are addressed in the answer.

    Uses an extended stop-word list (M-9) so common English function words
    don't inflate completeness scores when they trivially appear everywhere.
    """
    if not sub_queries:
        return 1.0

    answer_lower = answer.lower()
    answered = 0

    for sq in sub_queries:
        words = [
            w.lower().strip("?.,!;:\"'")
            for w in sq.split()
            if w.lower().strip("?.,!;:\"'") not in _STOP_WORDS and len(w) > 3
        ]
        if not words:
            answered += 1
            continue
        found = sum(1 for w in words if w in answer_lower)
        if found / len(words) >= 0.5:
            answered += 1

    score = answered / len(sub_queries)
    logger.debug(
        "verifier.completeness",
        sub_queries=len(sub_queries),
        answered=answered,
        score=round(score, 3),
    )
    return round(score, 3)


async def _check_consistency(
    query: str,
    context: str,
    settings: object,
) -> bool:
    """Run self-consistency check by generating N answers WITH context (M-4).

    Previously generated answers without context, causing false failures.
    Now includes the same retrieved context used by the main generator so
    the consistency samples are grounded in the same evidence.
    """
    from agent.prompts import GENERATOR_SYSTEM

    n = settings.consistency_samples
    hot_llm = _get_llm_hot()

    # Include context so samples are grounded (M-4 fix)
    # Truncate context to keep prompt cost reasonable
    context_excerpt = context[:3000] if context else ""
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Context:\n{context_excerpt}\n\n"
                f"Question: {query}\n\n"
                f"Answer based only on the context above:"
            ),
        },
    ]

    tasks = [hot_llm.ainvoke(messages) for _ in range(n)]
    try:
        responses = await asyncio.gather(*tasks)
        answers = [r.content.strip() for r in responses]
    except Exception as exc:
        logger.warning("verifier.consistency_generation_failed", error=str(exc))
        return True  # fail open

    judge_llm = _get_llm_zero()
    judge_chain = consistency_prompt | judge_llm
    answers_text = "\n\n---\n\n".join(
        f"Answer {i+1}:\n{a}" for i, a in enumerate(answers)
    )

    try:
        response = await judge_chain.ainvoke({"query": query, "answers": answers_text})
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        agreement: bool = bool(parsed.get("agreement", True))
        logger.info(
            "verifier.consistency_result",
            agreement=agreement,
            reasoning=parsed.get("reasoning", "")[:100],
        )
        return agreement
    except Exception as exc:
        logger.warning("verifier.consistency_judge_failed", error=str(exc))
        return True


def _verify_citations(state: AgentState) -> tuple[bool, list[str]]:
    """Cross-check citations against retrieved chunks."""
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
    """Hallucination guard: faithfulness + completeness + consistency + citations.

    Verification pipeline:
    1. Faithfulness — NLI model (D1) or LLM judge fallback
    2. Completeness — keyword overlap check across sub-queries (D2)
    3. Self-consistency — N parallel grounded generations agree (skipped on retry)
    4. Citation verification — all cited sources exist in retrieved chunks
    """
    settings = get_settings()
    answer = state.get("generated_answer", "")
    query = state["original_query"]
    sub_queries = state.get("sub_queries") or [query]
    retry_count = state.get("retry_count", 0)
    log = logger.bind(node="verifier", retry_count=retry_count)
    log.info("verifier_node.start")

    if not answer:
        return {
            **state,
            "faithfulness_score": 0.0,
            "consistency_passed": False,
            "completeness_score": 0.0,
            "final_answer": "",
        }

    context = _build_context_str(state)

    # ── D1: Faithfulness ──────────────────────────────────────────────────
    try:
        if settings.use_nli_faithfulness:
            loop = asyncio.get_running_loop()
            faithfulness_score = await loop.run_in_executor(
                None,
                _check_faithfulness_nli,
                answer,
                context,
                settings.nli_model,
            )
            log.info("verifier_node.faithfulness_nli", score=round(faithfulness_score, 3))
        else:
            faithfulness_score = await _check_faithfulness_llm(_get_llm_zero(), answer, context)
            log.info("verifier_node.faithfulness_llm", score=round(faithfulness_score, 3))
    except Exception as exc:
        log.warning("verifier_node.faithfulness_error", error=str(exc))
        faithfulness_score = 0.5

    # ── D2: Completeness ──────────────────────────────────────────────────
    completeness_score = _check_completeness(answer, sub_queries)
    log.info("verifier_node.completeness", score=completeness_score)

    # ── Self-consistency (skip on retries) ────────────────────────────────
    if retry_count == 0:
        consistency_passed = await _check_consistency(query, context, settings)
    else:
        log.info("verifier_node.consistency_skipped_on_retry")
        consistency_passed = True

    # ── Citation verification ─────────────────────────────────────────────
    citations_valid, hallucinated = _verify_citations(state)
    if not citations_valid:
        log.warning("verifier_node.hallucinated_citations", hallucinated=hallucinated)

    # ── Decision ──────────────────────────────────────────────────────────
    passes = faithfulness_score >= settings.faithfulness_threshold and consistency_passed
    final_answer = answer if passes else ""

    log.info(
        "verifier_node.done",
        faithfulness=round(faithfulness_score, 3),
        completeness=completeness_score,
        consistency=consistency_passed,
        citations_valid=citations_valid,
        passes=passes,
    )

    return {
        **state,
        "faithfulness_score": faithfulness_score,
        "completeness_score": completeness_score,
        "consistency_passed": consistency_passed,
        "final_answer": final_answer,
        "retry_count": retry_count,
    }
