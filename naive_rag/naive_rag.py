"""The fixed one-shot retrieve-then-read baseline used for evaluation."""

from adaptive import format_chunks, generate_answer
from tool.retriever import retrieve


def answer_question(question, knowledge_base, client, top_k=3):
    """Answer once from Top-K chunks retrieved with the original question.

    This intentionally excludes routing, rewriting, grading, iterative
    retrieval, web search, and answer-validation retries. It is the baseline
    against which the agentic retrieval strategy is measured.
    """
    docs = retrieve(question, knowledge_base, client, top_k=top_k)
    context = format_chunks(docs)
    answer = generate_answer(client, context, question)
    return {
        "question": question,
        "answer": answer,
        "docs": docs,
        "context": context,
        "retrieval_rounds": 1,
    }
