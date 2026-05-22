"""
agent/graph.py

LangGraph StateGraph — builds and compiles the RAG agent pipeline.

Graph topology:
    decomposer → retriever → generator → verifier
                     ↑                        |
                     └── (faith < thresh) ────┘
                                              |
                                             END

Node files imported here are the CANONICAL versions:
  - agent/nodes/retriever.py  (generator_node.py deleted — merged in)
  - agent/nodes/generator.py  (retriever_node.py deleted — merged in)

BUG-FIX (Issue 7):
  graph.py previously imported from generator.py (old, no multi-section fixes)
  while fixes existed in generator_node.py. Both files now merged into the
  canonical generator.py. Same for retriever.py / retriever_node.py.

BUG-FIX (Issue 5):
  All settings access uses snake_case (settings.faithfulness_threshold).
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from agent.nodes.decomposer import decomposer_node
from agent.nodes.generator import generator_node
from agent.nodes.retriever import retriever_node
from agent.nodes.verifier import verifier_node
from agent.state import AgentState
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

# Maximum re-retrieval attempts before accepting the best answer we have.
_MAX_RETRIES = 2


def _route_after_verifier(state: AgentState) -> str:
    """Conditional edge: decide what to do after verification.

    Routes:
    - END if verification passed or max retries reached.
    - "retriever" if faithfulness is below threshold and retries remain.

    Args:
        state: Current agent state after verifier node.

    Returns:
        Next node name or END sentinel.
    """
    settings = get_settings()
    # BUG-FIX (Issue 5): snake_case
    faithfulness = state.get("faithfulness_score", 0.0)
    consistency = state.get("consistency_passed", True)
    retry_count = state.get("retry_count", 0)
    final_answer = state.get("final_answer", "")

    # Already have a good answer
    if (
        final_answer
        and faithfulness >= settings.faithfulness_threshold
        and consistency
    ):
        logger.info(
            "graph.verification_passed",
            faithfulness=round(faithfulness, 3),
        )
        return END

    # Max retries reached — accept what we have
    if retry_count >= _MAX_RETRIES:
        logger.warning(
            "graph.max_retries_reached",
            retry_count=retry_count,
            faithfulness=round(faithfulness, 3),
        )
        return END

    # Re-retrieve with expanded context
    logger.info(
        "graph.verification_failed_retriggering",
        faithfulness=round(faithfulness, 3),
        retry_count=retry_count,
    )
    return "retriever"


async def _retriever_with_retry(state: AgentState) -> AgentState:
    """Wrapper around retriever_node that increments retry_count.

    Increments retry_count in state before calling retriever_node so
    the retriever knows this is a retry pass and can adjust top_k.

    Args:
        state: Current agent state.

    Returns:
        Updated state from retriever_node with incremented retry_count.
    """
    retry_count = state.get("retry_count", 0) + 1
    logger.info("graph.retry_retrieval", retry_count=retry_count)
    # Pass incremented retry_count into the retriever
    updated = await retriever_node({**state, "retry_count": retry_count})
    # Preserve retry_count in the returned state
    return {**updated, "retry_count": retry_count}


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph.

    All nodes are plain async functions that accept and return AgentState.
    The graph is compiled once at module load and reused across requests.

    Returns:
        Compiled StateGraph ready for ``ainvoke`` / ``astream``.
    """
    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("decomposer", decomposer_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("retriever_retry", _retriever_with_retry)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.set_entry_point("decomposer")
    graph.add_edge("decomposer", "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "verifier")

    # Conditional: verifier → END or re-retrieve
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {
            END: END,
            "retriever": "retriever_retry",
        },
    )
    graph.add_edge("retriever_retry", "generator")

    return graph.compile()


# Module-level compiled graph — import this in FastAPI routes.
rag_graph = build_graph()
