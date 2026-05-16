from __future__ import annotations

import asyncio
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
from openai import OpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._answer_relevance import answer_relevancy

from agent.graph import rag_graph
from agent.state import AgentState
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

QUERIES_PATH = Path(__file__).parent / "test_queries.json"
REPORTS_DIR = Path(__file__).parent / "reports"


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types returned by RAGAS."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_test_queries(path: Path = QUERIES_PATH) -> list[dict]:
    """Load golden queries from JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


async def run_single_query(question: str) -> dict[str, Any]:
    """Run a single question through the full RAG pipeline."""
    state: AgentState = {"original_query": question, "retry_count": 0}
    try:
        result = await rag_graph.ainvoke(state)
        answer = result.get("final_answer") or result.get("generated_answer", "")
        chunks = result.get("reranked_chunks") or result.get("retrieved_chunks") or []
        contexts = [c.text for c in chunks]
        faithfulness_score = result.get("faithfulness_score", 0.0)
    except Exception as exc:
        logger.error("Query failed during eval", question=question[:60], error=str(exc))
        answer = ""
        contexts = []
        faithfulness_score = 0.0

    return {
        "answer": answer,
        "contexts": contexts,
        "faithfulness_score": faithfulness_score,
    }


async def build_ragas_dataset(
    queries: list[dict],
    *,
    max_queries: int | None = None,
) -> tuple[Dataset, list[dict]]:
    """Run all queries and build a RAGAS-compatible Dataset."""
    if max_queries:
        queries = queries[:max_queries]

    log = logger.bind(total_queries=len(queries))
    log.info("Building RAGAS dataset")

    questions, answers, contexts, ground_truths = [], [], [], []
    raw_results: list[dict] = []

    for i, q in enumerate(queries, 1):
        log.info(f"Running query {i}/{len(queries)}", question=q["question"][:60])
        t0 = time.time()
        result = await run_single_query(q["question"])
        elapsed = round(time.time() - t0, 2)

        questions.append(q["question"])
        answers.append(result["answer"])
        contexts.append(
            result["contexts"] if result["contexts"] else ["No context retrieved"]
        )
        ground_truths.append(q["ground_truth"])

        raw_results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "answer": result["answer"],
            "contexts": result["contexts"],
            "faithfulness_score_agent": float(result["faithfulness_score"]),
            "elapsed_seconds": elapsed,
        })

        log.info(
            f"Query {i} complete",
            elapsed=elapsed,
            answer_length=len(result["answer"]),
        )
        await asyncio.sleep(1)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    log.info("RAGAS dataset built", rows=len(dataset))
    return dataset, raw_results


def score_dataset(dataset: Dataset) -> dict[str, float]:
    """Run RAGAS v0.4.3 evaluation.

    Uses old-style metric singletons (proper Metric instances).
    LangchainEmbeddingsWrapper needed for embed_query on answer_relevancy.
    All numpy scalars cast to Python float via _extract().
    """
    settings = get_settings()
    logger.info("Running RAGAS scoring")

    openai_client = OpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.llm_model, client=openai_client)
    embeddings = LangchainEmbeddingsWrapper(
        LCOpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
    )

    metrics = [faithfulness, context_precision, context_recall, answer_relevancy]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    def _extract(key: str) -> float:
        """Extract score as plain Python float — handles numpy scalars and lists."""
        val = result[key]
        if isinstance(val, list):
            valid = [float(v) for v in val if v is not None and str(v) != "nan"]
            return round(sum(valid) / len(valid), 4) if valid else 0.0
        try:
            return round(float(val), 4)
        except Exception:
            return 0.0

    scores = {
        "faithfulness": _extract("faithfulness"),
        "answer_relevancy": _extract("answer_relevancy"),
        "context_precision": _extract("context_precision"),
        "context_recall": _extract("context_recall"),
    }

    logger.info("RAGAS scoring complete", scores=scores)
    return scores


def save_report(
    scores: dict[str, float],
    raw_results: list[dict],
    *,
    reports_dir: Path = REPORTS_DIR,
) -> tuple[Path, Path]:
    """Save evaluation results as JSON and CSV.

    Uses _NumpyEncoder to handle any residual numpy types from RAGAS.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    baselines = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.70,
    }

    metric_results = {
        metric: {
            "score": float(score),
            "baseline": float(baselines[metric]),
            "passed": bool(score >= baselines[metric]),
        }
        for metric, score in scores.items()
    }

    report = {
        "timestamp": timestamp,
        "total_queries": len(raw_results),
        "metrics": metric_results,
        "overall_passed": bool(all(v["passed"] for v in metric_results.values())),
        "per_query": raw_results,
    }

    json_path = reports_dir / f"ragas_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)

    csv_path = reports_dir / f"ragas_report_{timestamp}.csv"
    if raw_results:
        fieldnames = list(raw_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_results:
                row_copy = {**row, "contexts": " | ".join(row.get("contexts", []))}
                writer.writerow(row_copy)

    logger.info("Reports saved", json=str(json_path), csv=str(csv_path))
    return json_path, csv_path


def print_summary(scores: dict[str, float], raw_results: list[dict]) -> None:
    """Print a human-readable evaluation summary."""
    baselines = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.70,
    }

    print("\n" + "=" * 60)
    print("  RAGAS EVALUATION REPORT")
    print("=" * 60)
    print(f"  Queries evaluated : {len(raw_results)}")
    print(f"  Timestamp         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    print(f"  {'Metric':<25} {'Score':>8}  {'Baseline':>8}  {'Status':>6}")
    print("-" * 60)

    all_passed = True
    for metric, score in scores.items():
        baseline = baselines[metric]
        passed = score >= baseline
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        print(f"  {metric:<25} {score:>8.4f}  {baseline:>8.2f}  {status:>6}")

    print("-" * 60)
    overall = "ALL PASSED" if all_passed else "SOME METRICS BELOW BASELINE"
    print(f"  Overall: {overall}")
    print("=" * 60 + "\n")

    # Note about Hindi PDF scores
    print("  Note: Low faithfulness/answer_relevancy scores are expected")
    print("  on Hindi PDFs — RAGAS judge is English-only. Scores will")
    print("  improve significantly with English PDFs in Phase 5.\n")


async def run_evaluation(
    *,
    max_queries: int | None = None,
    queries_path: Path = QUERIES_PATH,
) -> dict[str, float]:
    """Full evaluation pipeline: load → run → score → report."""
    queries = load_test_queries(queries_path)
    logger.info("Loaded queries", count=len(queries))

    dataset, raw_results = await build_ragas_dataset(
        queries, max_queries=max_queries
    )
    scores = score_dataset(dataset)
    json_path, csv_path = save_report(scores, raw_results)
    print_summary(scores, raw_results)

    print(f"  JSON report : {json_path}")
    print(f"  CSV report  : {csv_path}\n")

    return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Quick test with 3 queries"
    )
    args = parser.parse_args()

    max_q = 3 if args.smoke else args.max_queries
    asyncio.run(run_evaluation(max_queries=max_q))
