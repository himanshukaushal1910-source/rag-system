from __future__ import annotations

from typing import TypedDict

from retrieval.hybrid_retriever import RetrievedChunk


class Citation(TypedDict):
    """Structured citation extracted from a generated answer.

    Attributes:
        filename: Source PDF filename.
        page: 1-indexed page number.
        chunk_id: Qdrant point ID of the source chunk.
    """

    filename: str
    page: int
    chunk_id: str


class AgentState(TypedDict, total=False):
    """Shared state passed between all LangGraph nodes.

    Every field is optional (``total=False``) so nodes can be added to the
    graph without needing to initialise fields they don't touch. Nodes read
    what they need and write what they produce.

    Attributes:
        original_query: The raw query string from the client.
        sub_queries: Decomposed sub-queries produced by the decomposer node.
        retrieved_chunks: Raw chunks from hybrid retrieval (pre-rerank).
        reranked_chunks: Chunks after cross-encoder re-ranking.
        generated_answer: Raw answer from the generator node.
        citations: Structured citations extracted from the generated answer.
        faithfulness_score: RAGAS/judge faithfulness score (0.0 – 1.0).
        consistency_passed: Whether self-consistency check passed.
        final_answer: Verified answer returned to the client.
        error: Error message if any node fails fatally.
        retry_count: Number of re-retrieval attempts made by the verifier.
    """

    original_query: str
    sub_queries: list[str]
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    generated_answer: str
    citations: list[Citation]
    faithfulness_score: float
    consistency_passed: bool
    final_answer: str
    error: str | None
    retry_count: int
