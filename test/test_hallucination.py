"""Hallucination-check tests (no API calls: everything stubbed)."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adaptive

Q = "What is zero trust network access?"


def run_pipeline(judge_verdicts, generated=("answer v0", "answer v1", "answer v2", "answer v3")):
    """judge_verdicts: list of (verdict, reason) returned in order."""
    gen_calls = []

    def fake_generate(client, context, question, strict=False):
        gen_calls.append(strict)
        return generated[len(gen_calls) - 1]

    with (
        patch.object(adaptive, "dispatch", return_value={"route": "web_search"}),
        patch.object(adaptive, "web_search", return_value=[]),
        patch.object(adaptive, "generate_answer", side_effect=fake_generate),
        patch.object(
            adaptive, "check_hallucination", side_effect=list(judge_verdicts)
        ) as mock_judge,
        patch.object(
            adaptive, "answers_question", return_value=("yes", "stubbed")
        ),
    ):
        result = adaptive.answer_question(Q, knowledge_base=[], client=None)
    return result, gen_calls, mock_judge


# Case 1: grounded first try -> no retries.
result, gen_calls, mock_judge = run_pipeline([("grounded", "all claims cited")])
assert result["hallucination"] == "grounded" and result["hallucination_retries"] == 0
assert result["answer"] == "answer v0" and gen_calls == [False]
print("case 1 ok: grounded first try, no retry.")

# Case 2: hallucinated once, then grounded -> one strict retry.
result, gen_calls, mock_judge = run_pipeline(
    [("hallucinated", "unsupported date"), ("grounded", "fixed")]
)
assert result["hallucination"] == "grounded" and result["hallucination_retries"] == 1
assert result["answer"] == "answer v1" and gen_calls == [False, True]
print("case 2 ok: one strict retry, then grounded.")

# Case 3: always hallucinated -> 3 retries then refuse.
result, gen_calls, mock_judge = run_pipeline([("hallucinated", "nope")] * 4)
assert result["hallucination"] == "refused" and result["hallucination_retries"] == 3
assert result["answer"] == adaptive.REFUSAL and gen_calls == [False, True, True, True]
assert mock_judge.call_count == 4
print("case 3 ok: 3 retries exhausted -> refusal.")

print("ALL HALLUCINATION CHECKS PASSED")
