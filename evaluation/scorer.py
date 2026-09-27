"""Blind answer scoring and metric aggregation for the benchmark."""

from config import LLM_DEPLOYMENT


def score_answer(question, expected_facts, answer, client):
    """Blindly score an answer from 0 to 2 against predefined expected facts."""
    facts = "\n".join(f"- {fact}" for fact in expected_facts)
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=(
            "You are a blind evaluator. Score an answer against the question "
            "and required facts. You do not know which RAG system wrote it. "
            "Return exactly four lines: line 1 is only 0, 1, or 2; line 2 "
            "starts 'Matched:'; line 3 starts 'Missing:'; line 4 starts "
            "'Reason:'. Score 0 for incorrect/non-responsive, 1 for a "
            "partially correct answer missing important facts, and 2 for a "
            "correct answer covering the important facts."
        ),
        input=(
            f"Question:\n{question}\n\nExpected facts:\n{facts}\n\n"
            f"Answer to evaluate:\n{answer}"
        ),
    )
    lines = [
        line.strip()
        for line in (getattr(response, "output_text", "") or "").splitlines()
        if line.strip()
    ]
    try:
        score = int(lines[0])
    except (IndexError, ValueError):
        raise ValueError(f"Unexpected evaluator output: {lines!r}") from None
    if score not in (0, 1, 2):
        raise ValueError(f"Unexpected evaluator score: {score!r}")
    return {
        "score": score,
        "matched_facts": lines[1].removeprefix("Matched:").strip() if len(lines) > 1 else "",
        "missing_facts": lines[2].removeprefix("Missing:").strip() if len(lines) > 2 else "",
        "reason": lines[3].removeprefix("Reason:").strip() if len(lines) > 3 else "",
    }


def aggregate_scores(question_results):
    """Return the required normalized and relative-improvement metrics."""
    count = len(question_results)
    if not count:
        raise ValueError("Cannot aggregate an empty benchmark.")
    naive_total = sum(item["naive"]["evaluation"]["score"] for item in question_results)
    agentic_total = sum(item["agentic"]["evaluation"]["score"] for item in question_results)
    maximum = count * 2
    naive_mean = naive_total / maximum
    agentic_mean = agentic_total / maximum
    relative_improvement = (
        None if naive_mean == 0 else (agentic_mean - naive_mean) / naive_mean * 100
    )
    multi_hop = [item for item in question_results if item.get("type") == "multi_hop"]

    def average_latency(system):
        values = [item[system].get("latency_seconds") for item in question_results]
        values = [value for value in values if value is not None]
        return None if not values else sum(values) / len(values)

    def average_rounds():
        values = []
        for item in question_results:
            rounds = item["agentic"].get("retrieval_rounds", [])
            values.append(len(rounds) if isinstance(rounds, list) else rounds)
        return sum(values) / len(values)

    return {
        "question_count": count,
        "maximum_points": maximum,
        "naive": {"points": naive_total, "mean": naive_mean},
        "agentic": {"points": agentic_total, "mean": agentic_mean},
        "relative_improvement_percent": relative_improvement,
        "meets_twenty_percent_target": (
            relative_improvement is not None and relative_improvement >= 20
        ),
        "average_latency_seconds": {
            "naive": average_latency("naive"),
            "agentic": average_latency("agentic"),
        },
        "average_agentic_local_retrieval_rounds": average_rounds(),
        "multi_hop": {
            "question_count": len(multi_hop),
            "naive_points": sum(item["naive"]["evaluation"]["score"] for item in multi_hop),
            "agentic_points": sum(item["agentic"]["evaluation"]["score"] for item in multi_hop),
        },
    }
