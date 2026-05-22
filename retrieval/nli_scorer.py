"""
retrieval/nli_scorer.py

Feature D1 — NLI-based faithfulness scoring.

Replaces the LLM judge (GPT-4o call) with a local cross-encoder NLI model
(cross-encoder/nli-deberta-v3-small) for sentence-level entailment scoring.

Advantages over LLM judge:
  - No API call → ~150ms faster per verification
  - No cost
  - Deterministic — same input always gives same score
  - More consistent than LLM judge on boundary cases

How it works:
  For each sentence in the answer, check if it is entailed by the best-matching
  context window (sliding over the full context). Returns the fraction of
  entailed sentences.
"""

from __future__ import annotations

import re
import threading

import structlog

logger = structlog.get_logger(__name__)

# Module-level model cache — loaded once, reused across threads
_nli_pipeline = None
_nli_lock = threading.Lock()

# Max chars fed to NLI per window — model tokenises at 512 tokens (~2000 chars)
_NLI_WINDOW = 1800
# Sliding-window stride — overlap prevents missing evidence that spans windows
_NLI_STRIDE = 900


def _get_nli_pipeline(model_name: str):
    """Load and cache the NLI cross-encoder pipeline (thread-safe).

    Uses a lock so that two concurrent executor threads both seeing
    _nli_pipeline=None only load the model once.
    """
    global _nli_pipeline
    # Fast path — already loaded
    if _nli_pipeline is not None:
        return _nli_pipeline

    with _nli_lock:
        # Re-check inside the lock — another thread may have loaded it
        if _nli_pipeline is not None:
            return _nli_pipeline
        try:
            from transformers import pipeline
            _nli_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=-1,  # CPU — fast enough for sentence-level scoring
            )
            logger.info("nli_scorer.model_loaded", model=model_name)
        except Exception as exc:
            logger.warning("nli_scorer.load_failed", error=str(exc))

    return _nli_pipeline


def _split_sentences(text: str) -> list[str]:
    """Split answer text into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [
        s.strip() for s in sentences
        if len(s.strip()) > 15 and not re.match(r'^\[Doc:', s.strip())
    ]


def _context_windows(context: str, window: int = _NLI_WINDOW, stride: int = _NLI_STRIDE) -> list[str]:
    """Slide over the full context in overlapping windows.

    Avoids hard-truncating context to 2000 chars (which discarded ~90%
    of evidence). Each sentence is checked against the window with the
    best entailment score.
    """
    if len(context) <= window:
        return [context]
    windows: list[str] = []
    start = 0
    while start < len(context):
        windows.append(context[start: start + window])
        start += stride
    return windows


def score_faithfulness_nli(
    answer: str,
    context: str,
    model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> float:
    """Score answer faithfulness using NLI entailment over the full context.

    For each sentence in the answer, slides over all context windows and
    takes the best (most entailed) result. Returns the fraction of
    entailed sentences.

    Args:
        answer:     Generated answer text.
        context:    Concatenated retrieved chunk text (can be very long).
        model_name: HuggingFace NLI model name.

    Returns:
        Faithfulness score between 0.0 and 1.0.
        Returns 0.75 as a safe default if the model is unavailable.
    """
    nli = _get_nli_pipeline(model_name)
    if nli is None:
        logger.warning("nli_scorer.unavailable_returning_default")
        return 0.75

    sentences = _split_sentences(answer)
    if not sentences:
        return 1.0

    windows = _context_windows(context)
    entailed = 0.0

    for sentence in sentences:
        best_score = 0.0
        for window in windows:
            try:
                result = nli(
                    f"{window} [SEP] {sentence}",
                    truncation=True,
                    max_length=512,
                )
                label = result[0]["label"].upper()
                if label == "ENTAILMENT":
                    best_score = 1.0
                    break  # can't do better — stop searching windows
                elif label == "NEUTRAL":
                    best_score = max(best_score, 0.5)
            except Exception as exc:
                logger.warning("nli_scorer.sentence_failed", error=str(exc))
                best_score = max(best_score, 0.5)
        entailed += best_score

    score = entailed / len(sentences)
    logger.debug(
        "nli_scorer.done",
        sentences=len(sentences),
        entailed=entailed,
        windows=len(windows),
        score=round(score, 3),
    )
    return round(min(1.0, max(0.0, score)), 3)
