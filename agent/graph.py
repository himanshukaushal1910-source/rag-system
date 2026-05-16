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
    - ``"end"`` if verification passed or max retries reached.
    - ``"retriever"`` if faithfulness is below threshold and retries remain.

    Args:
        state: Current agent state after verifier node.

    Returns:
        Next node name or ``END``.
    """
    settings = get_settings()
    faithfulness = state.get("faithfulness_score", 0.0)
    consistency = state.get("consistency_passed", True)
    retry_count = state.get("retry_count", 0)
    final_answer = state.get("final_answer", "")

    # Already have a good answer
    if final_answer and faithfulness >= settings.faithfulness_threshold and consistency:
        logger.info("Verification passed — routing to END", faithfulness=round(faithfulness, 3))
        return END

    # Max retries reached — accept what we have
    if retry_count >= _MAX_RETRIES:
        logger.warning(
            "Max retries reached — routing to END with best available answer",
            retry_count=retry_count,
            faithfulness=round(faithfulness, 3),
        )
        return END

    # Re-retrieve with expanded context
    logger.info(
        "Verification failed — re-retrieving",
        faithfulness=round(faithfulness, 3),
        retry_count=retry_count,
    )
    return "retriever"


async def _retriever_with_retry(state: AgentState) -> AgentState:
    """Wrapper around retriever_node that increments retry_count.

    Also expands retrieval by passing a higher top_k on retries
    so the re-retrieval fetches more candidates.

    Args:
        state: Current agent state.

    Returns:
        Updated state from retriever_node with incremented retry_count.
    """
    retry_count = state.get("retry_count", 0) + 1
    logger.info("Re-retrieving", retry_count=retry_count)
    updated = await retriever_node({**state, "retry_count": retry_count})
    return {**updated, "retry_count": retry_count}


def build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph.

    Graph topology::

        decomposer → retriever → generator → verifier
                         ↑                        |
                         └── (if faith < thresh) ─┘
                                                  |
                                                 END

    Returns:
        Compiled :class:`StateGraph` ready for ``ainvoke`` / ``astream``.
    """
    graph = StateGraph(AgentState)

    # ------------------------------------------------------------------ #
    # Register nodes
    # ------------------------------------------------------------------ #
    graph.add_node("decomposer", decomposer_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("retriever_retry", _retriever_with_retry)

    # ------------------------------------------------------------------ #
    # Edges
    # ------------------------------------------------------------------ #
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
