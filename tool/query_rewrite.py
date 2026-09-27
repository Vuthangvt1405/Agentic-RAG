"""Retrieval-query rewriting for the local knowledge base."""

from config import LLM_DEPLOYMENT


def rewrite_query(question, previous_queries, evidence_gap, retrieval_round, client):
    """Return a focused, non-repeating query for a local retrieval round.

    The parser deliberately falls back to the original question. A malformed
    planner response must not prevent the RAG pipeline from answering.
    """
    prior = "\n".join(f"- {query}" for query in previous_queries) or "- none"
    gap = evidence_gap or "No evidence yet; identify the key concepts in the question."
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=(
            "You rewrite questions for semantic search over a local academic "
            "knowledge base. Return exactly two lines: line 1 is a concise "
            "search query; line 2 is a brief reason. Do not repeat a previous "
            "query. Preserve important entities and use terms likely to occur "
            "in source documents."
        ),
        input=(
            f"Original question:\n{question}\n\n"
            f"Retrieval round: {retrieval_round}\n"
            f"Evidence gap:\n{gap}\n\n"
            f"Previous queries:\n{prior}"
        ),
    )
    lines = [
        line.strip()
        for line in (getattr(response, "output_text", "") or "").splitlines()
        if line.strip()
    ]
    query = lines[0] if lines else question.strip()
    if not query or query.lower() in {item.lower() for item in previous_queries}:
        query = question.strip()
    reason = lines[1] if len(lines) > 1 else "Used the original question as a safe fallback."
    return query, reason
