"""
retrieval/step_back.py

Step-Back Prompting for improved retrieval coverage.

For a specific question like "What accuracy did BERT-large achieve on SST-2?",
step-back generates: "What are BERT performance benchmarks?"

Both the original and step-back query are used for retrieval. The step-back
query surfaces background/context chunks that the specific query might miss.

Reference: "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_STEP_BACK_SYSTEM_PROMPT = (
    "Given a specific question, generate a more general, abstract version that covers "
    "the broader topic. The abstract question should help retrieve background context "
    "and foundational information relevant to answering the specific question. "
    "Return ONLY the abstract question, no explanation."
)


async def generate_step_back_query(
    query: str,
    openai_client: Any,
) -> str:
    """Generate an abstract step-back query using GPT-4o-mini.

    A step-back query broadens the scope of retrieval by asking about the
    general topic rather than the specific detail.  When used alongside the
    original query, it surfaces background/foundational chunks that often
    provide essential context for precise factual questions.

    Examples:
      Specific: "What accuracy did BERT-large achieve on SST-2?"
      Step-back: "What are BERT performance benchmarks on NLP tasks?"

      Specific: "What is the learning rate used in GPT-3 pre-training?"
      Step-back: "What are the training hyperparameters for large language models?"

    Args:
        query:         Original user query string.
        openai_client: AsyncOpenAI client instance.

    Returns:
        Abstract step-back question string.  Returns an empty string on any
        error so callers can gracefully skip step-back retrieval.
    """
    log = logger.bind(query=query[:80])
    log.debug("step_back.generating")

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _STEP_BACK_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=128,
        )
        step_back = (response.choices[0].message.content or "").strip()

        if not step_back or len(step_back) < 5:
            log.warning("step_back.empty_response")
            return ""

        log.info("step_back.done", step_back=step_back[:100])
        return step_back

    except Exception as exc:
        log.warning("step_back.failed", error=str(exc))
        return ""


async def step_back_retrieve(
    original_query: str,
    openai_client: Any,
    retriever: Any,
    top_k: int,
) -> tuple[list[dict], str]:
    """Generate a step-back query and retrieve for both original and abstract queries.

    Runs both retrievals in parallel.  The step-back retrieval surfaces
    broader context chunks; the original retrieval keeps precision.

    Args:
        original_query: The user's specific question.
        openai_client:  AsyncOpenAI client for step-back generation.
        retriever:      HybridRetriever instance with a .retrieve() method.
        top_k:          Number of chunks per retrieval pass.

    Returns:
        Tuple of:
          - Merged deduplicated list of result dicts (original first, then step-back).
          - The step-back query string (empty string if generation failed).
    """
    log = logger.bind(query=original_query[:80], top_k=top_k)

    step_back_query = await generate_step_back_query(original_query, openai_client)

    if step_back_query:
        log.info("step_back.retrieving_both", step_back=step_back_query[:100])
        original_task = retriever.retrieve(original_query, top_k=top_k)
        step_back_task = retriever.retrieve(step_back_query, top_k=top_k)

        try:
            original_results, step_back_results = await asyncio.gather(
                original_task,
                step_back_task,
            )
        except Exception as exc:
            log.warning("step_back.parallel_retrieve_failed", error=str(exc))
            original_results = await retriever.retrieve(original_query, top_k=top_k)
            step_back_results = []
    else:
        log.info("step_back.skipping_no_query_generated")
        original_results = await retriever.retrieve(original_query, top_k=top_k)
        step_back_results = []

    # Merge: original results first, deduplicate step-back additions
    seen_ids: set[str] = {item["id"] for item in original_results}
    merged = list(original_results)

    for item in step_back_results:
        if item.get("id") not in seen_ids:
            merged.append(item)
            seen_ids.add(item["id"])

    log.info(
        "step_back.merge_done",
        original=len(original_results),
        step_back_additions=len(merged) - len(original_results),
        total=len(merged),
    )
    return merged, step_back_query
