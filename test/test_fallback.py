"""Fallback branching tests (no API calls: retrieve/grade/web/generate stubbed)."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adaptive

DOCS = [
    {"text": "Zero trust verifies every request", "source": "ztna.pdf", "score": 0.8},
    {"text": "Sourdough needs flour and water", "source": "baking.pdf", "score": 0.1},
]
WEB = [{"title": "ZTNA overview", "url": "https://example.com/ztna", "text": "ZTNA grants per-request access"}]
Q = "What is zero trust network access?"


def run_pipeline(route, grades, decision="answer"):
    with (
        patch.object(adaptive, "dispatch", return_value={"route": route}),
        patch.object(adaptive, "retrieve", return_value=DOCS),
        patch.object(adaptive, "grade_docs", return_value=grades),
        patch.object(adaptive, "rewrite_query", return_value=(Q, "stubbed rewrite")),
        patch.object(
            adaptive,
            "decide_next_retrieval",
            return_value=(decision, "local evidence incomplete"),
        ),
        patch.object(adaptive, "web_search", return_value=WEB) as mock_web,
        patch.object(adaptive, "generate_answer", return_value="mock answer") as mock_gen,
        patch.object(
            adaptive, "check_hallucination", return_value=("grounded", "stubbed")
        ),
        patch.object(
            adaptive, "answers_question", return_value=("yes", "stubbed")
        ),
    ):
        result = adaptive.answer_question(Q, knowledge_base=[], client=None)
    return result, mock_web, mock_gen


# Case 1: evidence controller says answer -> no web call, chunks-only context.
result, mock_web, mock_gen = run_pipeline("vectorstore", ["relevant", "relevant"])
assert result["route"] == "vectorstore"
assert result["web_results"] == []
mock_web.assert_not_called()
context = mock_gen.call_args[0][1]
assert "Zero trust verifies" in context and "example.com" not in context
print("case 1 ok: all relevant -> chunks only, no web call.")

# Case 2: evidence controller says web_search -> combined context.
result, mock_web, mock_gen = run_pipeline(
    "vectorstore", ["relevant", "irrelevant"], decision="web_search"
)
assert result["web_results"] == WEB
mock_web.assert_called_once_with("local evidence incomplete")
context = mock_gen.call_args[0][1]
assert "Zero trust verifies" in context and "https://example.com/ztna" in context
assert "Sourdough" not in context  # irrelevant chunk excluded
print("case 2 ok: fallback merges relevant chunk + web result.")

# Case 3: router says web_search -> straight to web, no retrieve/grade.
with (
    patch.object(adaptive, "dispatch", return_value={"route": "web_search"}),
    patch.object(adaptive, "retrieve") as mock_ret,
    patch.object(adaptive, "grade_docs") as mock_grade,
    patch.object(adaptive, "rewrite_query") as mock_rewrite,
    patch.object(adaptive, "decide_next_retrieval") as mock_decide,
    patch.object(adaptive, "web_search", return_value=WEB),
    patch.object(adaptive, "generate_answer", return_value="mock answer") as mock_gen,
    patch.object(
        adaptive, "check_hallucination", return_value=("grounded", "stubbed")
    ),
    patch.object(
        adaptive, "answers_question", return_value=("yes", "stubbed")
    ),
):
    result = adaptive.answer_question(Q, knowledge_base=[], client=None)
mock_ret.assert_not_called()
mock_grade.assert_not_called()
mock_rewrite.assert_not_called()
mock_decide.assert_not_called()
assert "https://example.com/ztna" in mock_gen.call_args[0][1]
print("case 3 ok: unrelated -> web only, retrieve/grade skipped.")

print("ALL FALLBACK CHECKS PASSED")
