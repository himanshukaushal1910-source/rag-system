"""
retrieval/rag_fusion.py

RAG Fusion: generate N paraphrase queries → retrieve for each → client-side RRF merge.

This is distinct from sub-query decomposition (which breaks a complex question into parts).
RAG Fusion generates PARAPHRASES of the same question to improve recall — different wording
might surface different chunks from the vector store.

Algorithm:
1. LLM generates N rephrased versions of the original query
2. Retrieve top_k chunks for each paraphrase using the existing hybrid retriever
3. Reciprocal Rank Fusion (RRF) merges results: score = sum(1/(k+rank_i)) for each document
4. Return top_k documents after RRF

Why RRF instead of score averaging:
- RRF is rank-based, not score-scale dependent
- Robust to varying score distributions across retrieval rounds
- A document appearing in position 1 across all paraphrases scores highest
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from retrieval.hybrid_retriever import HybridRetriever

logger = structlog.get_logger(__name__)


async def generate_rag_fusion_queries(
    query: str,
    n: int,
    openai_client: Any,
) -> list[str]:
    """Generate N paraphrase variants of query using GPT-4o-mini.

    Calls the OpenAI chat completions API and requests a JSON array of
    N rephrased versions of the same question.  Different wording helps
    surface chunks that wouldn't match the original phrasing.

    Args:
        query:         Original user query string.
        n:             Number of paraphrase variants to generate.
        openai_client: AsyncOpenAI client instance.

    Returns:
        List of paraphrase strings.  Falls back to [query] on any error
        so retrieval can always proceed.
    """
    log = logger.bind(query=query[:80], n=n)
    log.debug("rag_fusion.generating_variants")

    system_prompt = (
        "You are an expert at query reformulation. "
        "Given a question, generate alternative phrasings that ask for the same "
        "information but use different vocabulary, structure, or perspective. "
        "These variants will be used for document retrieval — diverse wording "
        "improves recall by matching different chunk styles."
    )

    user_prompt = (
        f"Generate exactly {n} alternative phrasings of the following question. "
        f"Each variant should seek the same answer but use different words. "
        f"Return ONLY a JSON array of strings, no explanation, no markdown fences.\n\n"
        f"Question: {query}\n\n"
        f"JSON array of {n} rephrased questions:"
    )

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        raw = (response.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        variants: list[str] = json.loads(raw)

        if not isinstance(variants, list) or not variants:
            raise ValueError(f"Expected non-empty list, got: {raw!r}")

        # Sanitise: keep only non-empty strings
        variants = [str(v).strip() for v in variants if str(v).strip()]

        if not variants:
            raise ValueError("All parsed variants were empty strings")

        log.info("rag_fusion.variants_generated", count=len(variants))
        return variants

    except Exception as exc:
        log.warning(
            "rag_fusion.generation_failed_using_original",
            error=str(exc),
        )
        return [query]


def rrf_merge(
    ranked_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF formula for each document d:
        score(d) = sum_i  1 / (k + rank_i(d))

    where rank_i(d) is the 1-based position of document d in list i
    (documents not present in a list contribute 0 to their sum).

    RRF is rank-based and therefore robust to differing score scales
    across retrieval rounds — a document ranked 1st in all lists beats
    one ranked 5th in all lists regardless of raw similarity scores.

    Args:
        ranked_lists: Each inner list is a ranked sequence of result dicts.
                      Every dict MUST contain an "id" key.
        k:            RRF constant (default 60 — standard in literature).
                      Higher k reduces the impact of top ranks.

    Returns:
        Flat list of result dicts sorted descending by "rag_fusion_score".
        Each dict carries all fields from its highest-scoring occurrence
        plus a new "rag_fusion_score" float key.
    """
    # accumulated RRF scores: id → float
    rrf_scores: dict[str, float] = {}
    # best payload / score seen for each id (from first encounter)
    best_item: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank_zero, item in enumerate(ranked_list):
            doc_id = item.get("id")
            if doc_id is None:
                continue
            rank_one = rank_zero + 1
            contribution = 1.0 / (k + rank_one)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + contribution

            # Keep metadata from first occurrence (preserves original payload)
            if doc_id not in best_item:
                best_item[doc_id] = item

    # Build output list with RRF score attached
    merged: list[dict] = []
    for doc_id, rrf_score in rrf_scores.items():
        item = dict(best_item[doc_id])  # shallow copy to avoid mutating originals
        item["rag_fusion_score"] = rrf_score
        merged.append(item)

    merged.sort(key=lambda x: x["rag_fusion_score"], reverse=True)

    logger.debug(
        "rrf_merge.done",
        input_lists=len(ranked_lists),
        unique_docs=len(merged),
        k=k,
    )
    return merged


async def rag_fusion_retrieve(
    original_query: str,
    variants: list[str],
    retriever: HybridRetriever,
    top_k: int,
) -> list[dict]:
    """Retrieve for all queries (original + variants) then RRF-merge results.

    Runs all retrievals in parallel via asyncio.gather.  Each query gets
    its own ranked list, which is then merged with client-side RRF so that
    documents consistently appearing near the top of many lists rank highest.

    Args:
        original_query: The user's original question (always included).
        variants:       Paraphrase variants from generate_rag_fusion_queries.
        retriever:      Initialised HybridRetriever instance.
        top_k:          Number of documents to return after RRF merge.

    Returns:
        Top-k result dicts sorted by rag_fusion_score (descending).
        Each dict has: id, score (original retrieval), payload, rag_fusion_score.
    """
    all_queries = [original_query] + variants
    log = logger.bind(
        original_query=original_query[:80],
        total_queries=len(all_queries),
        top_k=top_k,
    )
    log.info("rag_fusion.retrieve_start")

    # Retrieve for every query in parallel
    tasks = [
        retriever.retrieve(q, top_k=top_k)
        for q in all_queries
    ]

    per_query_results: list[list[dict]] = []
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            log.warning(
                "rag_fusion.query_failed",
                query=all_queries[i][:80],
                error=str(result),
            )
            per_query_results.append([])
        else:
            per_query_results.append(result)

    # Client-side RRF merge
    merged = rrf_merge(per_query_results)
    final = merged[:top_k]

    log.info(
        "rag_fusion.retrieve_done",
        unique_docs_before_trim=len(merged),
        returned=len(final),
    )
    return final
