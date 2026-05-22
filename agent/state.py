"""
agent/state.py

Shared state passed between all LangGraph nodes.

Added for chat continuation:
  chat_history:   List of previous Q&A pairs in this session.
                  Each entry: {"role": "user"|"assistant", "content": str}
  rewritten_query: Query after context-aware rewriting in decomposer.
  retrieved_images: Image dicts extracted by generator for API response.
  retrieved_tables: Table dicts extracted by generator for API response.
  completeness_score: Fraction of sub-queries answered (verifier D2).
"""

from __future__ import annotations

from typing import TypedDict

from retrieval.hybrid_retriever import RetrievedChunk


class Citation(TypedDict):
    """Structured citation extracted from a generated answer."""
    filename: str
    page: int
    chunk_id: str


class ChatMessage(TypedDict):
    """A single message in the conversation history."""
    role: str       # "user" or "assistant"
    content: str    # query text or answer text


class AgentState(TypedDict, total=False):
    """Shared state passed between all LangGraph nodes.

    Every field is optional (total=False) so nodes can be added without
    initialising fields they don't touch.

    Attributes:
        original_query:    Raw query string from the client.
        rewritten_query:   Query after context-aware rewriting.
        sub_queries:       Decomposed sub-queries from decomposer node.
        chat_history:      Previous Q&A pairs for conversation continuity.
                           Each entry: {"role": "user"|"assistant", "content": str}
        retrieved_chunks:  Raw chunks from hybrid retrieval (pre-rerank).
        reranked_chunks:   Chunks after cross-encoder reranking.
        generated_answer:  Raw answer from generator node.
        citations:         Structured citations from generated answer.
        retrieved_images:  Image dicts for API response (generator).
        retrieved_tables:  Table dicts for API response (generator).
        faithfulness_score: Faithfulness score 0.0–1.0 (verifier).
        consistency_passed: Whether self-consistency check passed.
        completeness_score: Fraction of sub-queries answered (verifier).
        final_answer:      Verified answer returned to client.
        error:             Error message if any node fails fatally.
        retry_count:       Number of re-retrieval attempts by verifier.
    """

    original_query: str
    rewritten_query: str
    sub_queries: list[str]
    chat_history: list[ChatMessage]
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    generated_answer: str
    citations: list[Citation]
    retrieved_images: list[dict]
    retrieved_tables: list[dict]
    faithfulness_score: float
    consistency_passed: bool
    completeness_score: float
    final_answer: str
    error: str | None
    retry_count: int
    # Advanced retrieval metadata
    query_type: str          # factual | analytical | comparative | visual | table | code
    step_back_query: str     # abstract step-back version of query (may be empty)
    fusion_queries: list[str]  # paraphrase variants used in RAG Fusion
