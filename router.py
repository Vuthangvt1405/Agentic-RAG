"""Routing stage of the adaptive RAG flow (see asset/langgraph_adaptive_rag.png).

Plain Python, no orchestration framework.

Question -> route_question -> [related to index] retrieve | [unrelated] web_search
"""

from typing import Literal

from config import LLM_DEPLOYMENT, build_client

Route = Literal["vectorstore", "web_search"]

# What the local index covers (the 5 theses in data/).
INDEX_SCOPE = (
    "Zero Trust architecture and access (ZTNA/ZTA), micro-segmentation, "
    "software-defined networking, and network security / cybersecurity "
    "thesis topics."
)


def route_question(question):
    """Classify a question as related to the index or not."""
    client = build_client()
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=(
            "You are a query router. Reply with exactly one word, "
            "either 'vectorstore' or 'web_search'. "
            f"Reply 'vectorstore' if the question is about: {INDEX_SCOPE} "
            "Reply 'web_search' otherwise."
        ),
        input=question,
    )
    route = (getattr(response, "output_text", "") or "").strip().lower()
    if route not in ("vectorstore", "web_search"):
        raise ValueError(f"Unexpected router output: {route!r}")
    return route


def retrieve(question):
    """Stub for the Retrieve Documents branch (next stage)."""
    raise NotImplementedError("Retrieve branch not wired yet.")


def web_search_branch(question):
    """Stub for the Web Search branch (next stage)."""
    raise NotImplementedError("Web Search branch not wired yet.")


def dispatch(question):
    """Route a question to its branch. Returns {"question", "route"}."""
    route = route_question(question)
    return {"question": question, "route": route}
