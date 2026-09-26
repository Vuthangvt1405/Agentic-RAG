"""Grade Documents stage: one relevance verdict per retrieved chunk.

Uses the LLM as judge. All chunks are graded in a single call;
each verdict is 'relevant' or 'irrelevant'.
"""

from config import LLM_DEPLOYMENT

VALID_GRADES = ("relevant", "irrelevant")


def grade_docs(question, docs, client):
    """Grade each doc against the question. Returns [grade] aligned with docs."""
    numbered = "\n\n".join(
        f"Document {i + 1}:\n{doc['text']}" for i, doc in enumerate(docs)
    )
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=(
            "You grade retrieved documents. A document is 'relevant' only if "
            "it contains facts needed to answer the question; otherwise it "
            f"is 'irrelevant'. Reply with exactly {len(docs)} lines, one per "
            "document in order, each line exactly one word: "
            "'relevant' or 'irrelevant'. No other text."
        ),
        input=f"Question:\n{question}\n\n{numbered}",
    )
    grades = [
        line.strip().lower()
        for line in (getattr(response, "output_text", "") or "").splitlines()
        if line.strip()
    ]
    if len(grades) != len(docs) or any(g not in VALID_GRADES for g in grades):
        raise ValueError(f"Unexpected grader output: {grades!r}")
    return grades


def any_irrelevant(grades):
    return any(g == "irrelevant" for g in grades)
