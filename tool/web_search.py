"""Web Search branch of the adaptive RAG flow (see asset/langgraph_adaptive_rag.png).

Uses the Exa Search API. Returns plain dicts so the branch
stays decoupled from the Exa SDK types.
"""

from config import EXA_API_KEY, EXA_MAX_RESULTS


def build_exa_client():
    """Build an Exa client from the EXA_API_KEY env var."""
    from exa_py import Exa

    if not EXA_API_KEY:
        raise RuntimeError(
            "EXA_API_KEY is not set. Add it to .env (see .env.example)."
        )
    return Exa(api_key=EXA_API_KEY)


def web_search(query, max_results=None):
    """Run an Exa search and return [{title, url, text}]."""
    client = build_exa_client()
    response = client.search(
        query,
        num_results=max_results or EXA_MAX_RESULTS,
        contents={"text": True},
    )
    results = []
    for item in response.results:
        results.append(
            {
                "title": getattr(item, "title", ""),
                "url": getattr(item, "url", ""),
                "text": getattr(item, "text", "")
                or " ".join(getattr(item, "highlights", None) or []),
            }
        )
    return results
