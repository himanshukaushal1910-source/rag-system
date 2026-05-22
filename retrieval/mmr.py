"""
retrieval/mmr.py

Feature B3 — Maximal Marginal Relevance (MMR) diversification.

After reranking, MMR selects chunks that are both relevant to the
query AND diverse from each other. This prevents the generator from
receiving 3 near-identical chunks while a critical different chunk
is excluded from the top-k.

Algorithm:
  1. Start with the highest-scored chunk
  2. Each subsequent selection maximises:
     lambda * relevance - (1-lambda) * max_similarity_to_selected
  3. Repeat until top_k chunks selected

lambda=1.0 → pure relevance (same as no MMR)
lambda=0.5 → balanced (default)
lambda=0.0 → pure diversity
"""

from __future__ import annotations

import numpy as np
import structlog

from retrieval.hybrid_retriever import RetrievedChunk

logger = structlog.get_logger(__name__)


def mmr_select(
    chunks: list[RetrievedChunk],
    embeddings: list[list[float]],
    top_k: int,
    lambda_param: float = 0.5,
) -> list[RetrievedChunk]:
    """Select top_k chunks using Maximal Marginal Relevance.

    Uses numpy for O(n×d) vectorised similarity — avoids O(n²×d) pure-Python loop.
    Requires pre-computed embeddings for each chunk (same order as chunks).
    Falls back to returning top_k by score if embeddings are unavailable.

    Args:
        chunks:       Reranked chunks to select from.
        embeddings:   Dense embedding for each chunk (same order).
        top_k:        Number of chunks to select.
        lambda_param: Trade-off between relevance and diversity (0–1).

    Returns:
        Selected list of up to top_k RetrievedChunk objects.
    """
    if not chunks:
        return []

    if not embeddings or len(embeddings) != len(chunks):
        logger.warning("mmr.no_embeddings_falling_back", chunks=len(chunks))
        return chunks[:top_k]

    top_k = min(top_k, len(chunks))
    n = len(chunks)

    # Build (n, d) matrix and L2-normalise rows for fast cosine via dot product
    emb_matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb_matrix = emb_matrix / norms  # unit vectors → dot = cosine

    # Normalise reranker scores to [0, 1]
    scores = np.array([c.score for c in chunks], dtype=np.float32)
    score_min, score_max = scores.min(), scores.max()
    score_range = score_max - score_min
    if score_range > 0:
        norm_scores = (scores - score_min) / score_range
    else:
        norm_scores = np.ones(n, dtype=np.float32)

    selected_indices: list[int] = []
    candidate_mask = np.ones(n, dtype=bool)

    for _ in range(top_k):
        if not candidate_mask.any():
            break

        if not selected_indices:
            # First pick: highest relevance
            candidates = np.where(candidate_mask)[0]
            best_idx = int(candidates[np.argmax(norm_scores[candidates])])
        else:
            # Similarity of all candidates to all selected — vectorised
            sel_matrix = emb_matrix[selected_indices]          # (k, d)
            cand_indices = np.where(candidate_mask)[0]
            cand_matrix = emb_matrix[cand_indices]             # (c, d)
            sim_matrix = cand_matrix @ sel_matrix.T            # (c, k)
            max_sim = sim_matrix.max(axis=1)                   # (c,)
            mmr_scores = (
                lambda_param * norm_scores[cand_indices]
                - (1 - lambda_param) * max_sim
            )
            best_local = int(np.argmax(mmr_scores))
            best_idx = int(cand_indices[best_local])

        selected_indices.append(best_idx)
        candidate_mask[best_idx] = False

    logger.debug(
        "mmr.selected",
        selected=len(selected_indices),
        from_total=n,
        lambda_param=lambda_param,
    )

    return [chunks[i] for i in selected_indices]
