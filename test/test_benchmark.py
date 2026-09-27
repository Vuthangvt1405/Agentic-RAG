"""Benchmark baseline, aggregation, and runner checks (no API calls)."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.run_benchmark import load_questions, run_benchmark
from evaluation.scorer import aggregate_scores
from naive_rag import naive_rag


# Naïve RAG must use the original question for exactly one fixed Top-K retrieval.
docs = [{"text": "A retrieved fact", "source": "source.pdf", "score": 0.9}]
with (
    patch.object(naive_rag, "retrieve", return_value=docs) as mock_retrieve,
    patch.object(naive_rag, "generate_answer", return_value="baseline answer") as mock_generate,
):
    baseline = naive_rag.answer_question("Original question", knowledge_base=[], client="client")
mock_retrieve.assert_called_once_with("Original question", [], "client", top_k=3)
assert baseline["retrieval_rounds"] == 1
assert baseline["docs"] == docs
assert mock_generate.call_count == 1
print("baseline case ok: one original-query Top-K retrieval and one generation.")


# The report must use the stated relative-improvement formula.
scored = [
    {"naive": {"evaluation": {"score": 1}}, "agentic": {"evaluation": {"score": 2}}},
    {"naive": {"evaluation": {"score": 1}}, "agentic": {"evaluation": {"score": 1}}},
]
summary = aggregate_scores(scored)
assert summary["naive"]["mean"] == 0.5
assert summary["agentic"]["mean"] == 0.75
assert summary["relative_improvement_percent"] == 50.0
assert summary["meets_twenty_percent_target"] is True
print("aggregation case ok: 50% relative improvement is calculated correctly.")


# The production benchmark is deliberately fixed at 20 questions.
questions = load_questions()
assert len(questions) == 20
assert sum(item.get("minimum_retrieval_rounds", 1) >= 2 for item in questions) >= 1
print("suite case ok: exactly 20 questions and multi-hop coverage present.")


def fake_naive(question, knowledge_base, client):
    return {"answer": f"naive {question}", "docs": [], "context": "", "retrieval_rounds": 1}


def fake_agentic(question, knowledge_base, client):
    return {"answer": f"agentic {question}", "docs": [], "context": "", "retrieval_rounds": []}


def fake_scorer(question, expected_facts, answer, client):
    return {"score": 2 if answer.startswith("agentic") else 1, "matched_facts": "", "missing_facts": "", "reason": "stubbed"}


tiny_suite = [
    {"id": "demo", "question": "demo question", "type": "direct_fact", "expected_facts": ["fact"], "expected_sources": []}
]
results, tiny_summary = run_benchmark(
    tiny_suite, [], None, naive_runner=fake_naive, agentic_runner=fake_agentic, scorer=fake_scorer
)
assert results[0]["naive"]["evaluation"]["score"] == 1
assert results[0]["agentic"]["evaluation"]["score"] == 2
assert tiny_summary["relative_improvement_percent"] == 100.0
print("runner case ok: both systems are scored blindly and stored together.")

print("ALL BENCHMARK CHECKS PASSED")
