"""Answers-question check: does the answer actually resolve the user's question?

This is sufficiency (useful?), not faithfulness (honest?) — that is the
hallucination judge's job. A grounded "I don't know" still fails here.
"""

from config import LLM_DEPLOYMENT

VALID_VERDICTS = ("yes", "no")


def answers_question(question, answer, client):
    """Return (verdict, reason): does the answer resolve the question?"""
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=(
            "You judge whether an answer resolves the user's question. Reply "
            "'yes' only if the answer directly addresses what was asked with "
            "a substantive response. Replies like 'I don't know', partial or "
            "tangential information, or answers to a different question are "
            "'no'. Reply with exactly two lines: line 1 is one word, 'yes' "
            "or 'no'; line 2 is a brief reason. No other text."
        ),
        input=f"Question:\n{question}\n\nAnswer:\n{answer}",
    )
    lines = [
        line.strip()
        for line in (getattr(response, "output_text", "") or "").splitlines()
        if line.strip()
    ]
    if len(lines) < 2 or lines[0].lower() not in VALID_VERDICTS:
        raise ValueError(f"Unexpected answer-check output: {lines!r}")
    return lines[0].lower(), lines[1]
