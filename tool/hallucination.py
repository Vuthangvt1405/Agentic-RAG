"""Hallucination check: is the generated answer grounded in its context?

The judge requires BOTH:
  1. Every factual claim in the answer is supported by the context.
  2. The overall answer gist stays consistent with the context.
"""

from config import LLM_DEPLOYMENT

VALID_VERDICTS = ("grounded", "hallucinated")


def check_hallucination(answer, context, client):
    """Return (verdict, reason) for an answer against its context."""
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=(
            "You judge whether an answer is hallucinated. An answer is "
            "'grounded' only if BOTH hold: (1) every factual claim in the "
            "answer is supported by the context, and (2) the overall gist "
            "of the answer stays consistent with the context. Otherwise it "
            "is 'hallucinated'. Reply with exactly two lines: line 1 is one "
            "word, 'grounded' or 'hallucinated'; line 2 is a brief reason. "
            "No other text."
        ),
        input=f"Context:\n{context}\n\nAnswer:\n{answer}",
    )
    lines = [
        line.strip()
        for line in (getattr(response, "output_text", "") or "").splitlines()
        if line.strip()
    ]
    if len(lines) < 2 or lines[0].lower() not in VALID_VERDICTS:
        raise ValueError(f"Unexpected hallucination-judge output: {lines!r}")
    return lines[0].lower(), lines[1]
