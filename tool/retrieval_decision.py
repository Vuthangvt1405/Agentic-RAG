"""Evidence-based controller for iterative local retrieval."""

from config import LLM_DEPLOYMENT

VALID_ACTIONS = ("answer", "retrieve_again", "web_search")


def decide_next_retrieval(question, relevant_docs, latest_docs, latest_grades, retrieval_round, client):
    """Return (action, evidence_gap) after a retrieval-and-grading round."""
    evidence = "\n\n".join(
        f"[source: {doc['source']}]\n{doc['text']}" for doc in relevant_docs
    ) or "No relevant local evidence found."
    latest = "\n".join(
        f"{grade}: {doc['text']}" for doc, grade in zip(latest_docs, latest_grades)
    ) or "No new documents retrieved."
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=(
            "You control evidence gathering for an agentic RAG system. Decide "
            "whether the available LOCAL evidence can answer the question. "
            "Return exactly two lines: line 1 is one of 'answer', "
            "'retrieve_again', or 'web_search'; line 2 states the missing fact "
            "or relationship, or says why the evidence is sufficient. Choose "
            "retrieve_again only if a focused follow-up query could plausibly "
            "find the missing evidence in the local corpus."
        ),
        input=(
            f"Question:\n{question}\n\nRetrieval round: {retrieval_round}\n\n"
            f"Relevant evidence collected:\n{evidence}\n\n"
            f"Latest retrieval grades:\n{latest}"
        ),
    )
    lines = [
        line.strip()
        for line in (getattr(response, "output_text", "") or "").splitlines()
        if line.strip()
    ]
    action = lines[0].lower() if lines else "web_search"
    if action not in VALID_ACTIONS:
        action = "web_search"
    gap = lines[1] if len(lines) > 1 else "Local evidence was insufficient."
    return action, gap
