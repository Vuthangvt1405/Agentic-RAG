"""Answers-question check tests (no API calls: everything stubbed)."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import adaptive

Q = "What is zero trust network access?"
WEB = [{"title": "ZTNA overview", "url": "https://example.com/ztna", "text": "ZTNA grants per-request access"}]


def run_pipeline(check_verdicts, generated=("answer v0", "answer v1", "answer v2")):
    gen_calls = []

    def fake_generate(client, context, question, strict=False):
        gen_calls.append((context, strict))
        return generated[len(gen_calls) - 1]

    with (
        patch.object(adaptive, "dispatch", return_value={"route": "web_search"}),
        patch.object(adaptive, "web_search", return_value=WEB) as mock_web,
        patch.object(adaptive, "generate_answer", side_effect=fake_generate),
        patch.object(
            adaptive, "check_hallucination", return_value=("grounded", "stubbed")
        ),
        patch.object(
            adaptive, "answers_question", side_effect=list(check_verdicts)
        ) as mock_check,
    ):
        result = adaptive.answer_question(Q, knowledge_base=[], client=None)
    return result, gen_calls, mock_web, mock_check


# Case 1: answers question first try -> no extra web round.
result, gen_calls, mock_web, mock_check = run_pipeline([("yes", "direct answer")])
assert result["answers_question"] == "yes" and result["answers_question_rounds"] == 0
assert result["answer"] == "answer v0" and mock_web.call_count == 1  # initial route only
print("case 1 ok: answered first try.")

# Case 2: 'no' once -> one extra web round, then 'yes'.
result, gen_calls, mock_web, mock_check = run_pipeline(
    [("no", "too vague"), ("yes", "now substantive")]
)
assert result["answers_question"] == "yes" and result["answers_question_rounds"] == 1
assert mock_web.call_count == 2
assert "https://example.com/ztna" in gen_calls[1][0]  # web text in regen context
assert result["answer"] == "answer v1"
print("case 2 ok: one extra web round, then answered.")

# Case 3: always 'no' -> capped at 2 extra rounds, best-effort answer kept.
result, gen_calls, mock_web, mock_check = run_pipeline([("no", "nope")] * 3)
assert result["answers_question"] == "no" and result["answers_question_rounds"] == 2
assert mock_web.call_count == 3 and len(gen_calls) == 3
print("case 3 ok: capped at 2 extra rounds, best effort kept.")

print("ALL ANSWER-CHECK TESTS PASSED")
