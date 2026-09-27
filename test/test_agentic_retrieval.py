"""Agentic retrieval-loop tests (no API calls)."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adaptive

Q = "How does micro-segmentation support a Zero Trust migration?"
ROUND_ONE = [
    {"text": "Zero Trust verifies every access request.", "source": "ztna.pdf", "score": 0.9},
]
ROUND_TWO = [
    {"text": "Micro-segmentation limits lateral movement during a Zero Trust migration.", "source": "migration.pdf", "score": 0.8},
]


with (
    patch.object(adaptive, "dispatch", return_value={"route": "vectorstore"}),
    patch.object(adaptive, "rewrite_query", side_effect=[
        ("Zero Trust migration principles", "find the overall migration guidance"),
        ("micro-segmentation in Zero Trust migration", "find the missing relationship"),
    ]) as mock_rewrite,
    patch.object(adaptive, "retrieve", side_effect=[ROUND_ONE, ROUND_TWO]) as mock_retrieve,
    patch.object(adaptive, "grade_docs", side_effect=[["relevant"], ["relevant"]]),
    patch.object(adaptive, "decide_next_retrieval", side_effect=[
        ("retrieve_again", "How micro-segmentation supports the migration is missing."),
        ("answer", "Both the migration and micro-segmentation relationship are supported."),
    ]),
    patch.object(adaptive, "generate_answer", return_value="grounded answer"),
    patch.object(adaptive, "check_hallucination", return_value=("grounded", "stubbed")),
    patch.object(adaptive, "answers_question", return_value=("yes", "stubbed")),
    patch.object(adaptive, "web_search") as mock_web,
):
    result = adaptive.answer_question(Q, knowledge_base=[], client=None)

assert mock_rewrite.call_count == 2 and mock_retrieve.call_count == 2
assert [round_["query"] for round_ in result["retrieval_rounds"]] == [
    "Zero Trust migration principles",
    "micro-segmentation in Zero Trust migration",
]
assert result["retrieval_rounds"][0]["decision"] == "retrieve_again"
assert result["retrieval_rounds"][1]["decision"] == "answer"
assert "Micro-segmentation limits lateral movement" in result["context"]
mock_web.assert_not_called()
print("multi-hop case ok: agent rewrites and retrieves a second local evidence round.")


# The cap protects against an unbounded agent loop and moves to web fallback.
with (
    patch.object(adaptive, "dispatch", return_value={"route": "vectorstore"}),
    patch.object(adaptive, "rewrite_query", side_effect=[("first", ""), ("second", "")]),
    patch.object(adaptive, "retrieve", side_effect=[ROUND_ONE, ROUND_TWO]),
    patch.object(adaptive, "grade_docs", side_effect=[["relevant"], ["relevant"]]),
    patch.object(adaptive, "decide_next_retrieval", return_value=("retrieve_again", "Need more evidence.")),
    patch.object(adaptive, "web_search", return_value=[] ) as mock_web,
    patch.object(adaptive, "generate_answer", return_value="grounded answer"),
    patch.object(adaptive, "check_hallucination", return_value=("grounded", "stubbed")),
    patch.object(adaptive, "answers_question", return_value=("yes", "stubbed")),
):
    capped = adaptive.answer_question(Q, knowledge_base=[], client=None)

assert len(capped["retrieval_rounds"]) == adaptive.MAX_LOCAL_RETRIEVAL_ROUNDS
assert capped["retrieval_rounds"][-1]["decision"] == "web_search"
mock_web.assert_called_once()
print("cap case ok: two local rounds then controlled web fallback.")

print("ALL AGENTIC-RETRIEVAL CHECKS PASSED")
