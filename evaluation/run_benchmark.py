"""Run the 20-question naïve-RAG versus agentic-RAG evaluation."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaptive import answer_question as agentic_answer_question
from config import build_client
from evaluation.scorer import aggregate_scores, score_answer
from naive_rag.naive_rag import answer_question as naive_answer_question
from tool.knowledge_base import build_chunks, ensure_embeddings, load_documents

QUESTIONS_PATH = ROOT / "questions" / "agentic_rag_eval_20.json"
RESULTS_DIR = ROOT / "evaluation" / "results"


def load_questions(path=QUESTIONS_PATH):
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(questions) != 20:
        raise ValueError(f"Benchmark must contain exactly 20 questions, found {len(questions)}.")
    required = {"id", "question", "type", "expected_facts", "expected_sources"}
    for item in questions:
        missing = required - item.keys()
        if missing or not item["expected_facts"]:
            raise ValueError(f"Invalid benchmark item {item.get('id')!r}: missing {missing} or facts.")
    return questions


def timed_call(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, round(time.perf_counter() - start, 3)


def run_benchmark(questions, knowledge_base, client, naive_runner=naive_answer_question,
                  agentic_runner=agentic_answer_question, scorer=score_answer):
    """Run both systems and score each answer without passing system identity."""
    results = []
    for item in questions:
        naive, naive_seconds = timed_call(naive_runner, item["question"], knowledge_base, client)
        agentic, agentic_seconds = timed_call(agentic_runner, item["question"], knowledge_base, client)
        naive["evaluation"] = scorer(item["question"], item["expected_facts"], naive["answer"], client)
        agentic["evaluation"] = scorer(item["question"], item["expected_facts"], agentic["answer"], client)
        results.append({
            "id": item["id"],
            "question": item["question"],
            "type": item["type"],
            "expected_facts": item["expected_facts"],
            "expected_sources": item["expected_sources"],
            "minimum_retrieval_rounds": item.get("minimum_retrieval_rounds", 1),
            "naive": {**naive, "latency_seconds": naive_seconds},
            "agentic": {**agentic, "latency_seconds": agentic_seconds},
        })
    return results, aggregate_scores(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    client = build_client()
    knowledge_base = ensure_embeddings(build_chunks(load_documents()), client)
    results, summary = run_benchmark(questions, knowledge_base, client)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "20-question agentic RAG vs one-shot naïve RAG",
        "results": results,
        "summary": summary,
    }
    output = args.output or RESULTS_DIR / f"benchmark-{datetime.now():%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed benchmark report to {output}")


if __name__ == "__main__":
    main()
