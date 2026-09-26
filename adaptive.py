"""Adaptive RAG pipeline (see asset/workflow.png), plain Python.

Question -> Routing -> vectorstore: Retrieve -> Grade ->
    all relevant: Generate Answer (chunks only)
    any irrelevant: Web Search -> Generate Answer (relevant chunks + web)
  -> web_search route: Web Search -> Generate Answer (web only)
"""

from config import LLM_DEPLOYMENT
from router import dispatch
from tool.answer_check import answers_question
from tool.grader import any_irrelevant, grade_docs
from tool.hallucination import check_hallucination
from tool.loading import stage
from tool.retriever import retrieve
from tool.web_search import web_search

REFUSAL = "I don't know based on the provided documents."
MAX_HALLUCINATION_RETRIES = 3
MAX_ANSWER_ROUNDS = 2


def format_chunks(docs):
    return "\n\n".join(
        f"[source: {doc['source']}]\n{doc['text']}" for doc in docs
    )


def format_web(results):
    return "\n\n".join(
        f"[source: {item['url']}]\n{item['title']}\n{item['text']}"
        for item in results
    )


def generate_answer(client, context, question, strict=False):
    instructions = (
        "Answer ONLY using the provided context. "
        "If the answer cannot be found in the context, "
        "say 'I don't know based on the provided documents.'"
    )
    if strict:
        instructions += (
            " Your previous answer contained unsupported claims. "
            "This time, use only facts stated word-for-word in the context. "
            "When unsure, refuse instead of guessing."
        )
    response = client.responses.create(
        model=LLM_DEPLOYMENT,
        instructions=instructions,
        input=f"Context:\n{context}\n\nQuestion:\n{question}",
    )
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    return response.output[0].content[0].text


def generate_grounded(client, context, question):
    """Generate + hallucination retries. Returns (answer, verdict, reason, retries, refused)."""
    with stage("Generating answer"):
        answer = generate_answer(client, context, question)

    hallucination, reason, retries, refused = "grounded", "", 0, False
    while True:
        with stage("Checking for hallucinations"):
            hallucination, reason = check_hallucination(answer, context, client)
        if hallucination == "grounded" or retries == MAX_HALLUCINATION_RETRIES:
            break
        retries += 1
        with stage(f"Regenerating answer (attempt {retries})"):
            answer = generate_answer(client, context, question, strict=True)

    if hallucination == "hallucinated":
        answer = REFUSAL
        refused = True
    return answer, hallucination, reason, retries, refused


def answer_question(question, knowledge_base, client, top_k=3):
    """Run the full pipeline. Returns {route, docs, grades, web_results, context, answer}."""
    with stage("Routing question"):
        route = dispatch(question)["route"]

    docs, grades, web_results = [], [], []

    if route == "web_search":
        with stage("Searching the web"):
            web_results = web_search(question)
        context = format_web(web_results)
    else:
        with stage("Retrieving documents"):
            docs = retrieve(question, knowledge_base, client, top_k=top_k)
        with stage("Grading documents"):
            grades = grade_docs(question, docs, client)
        relevant = [d for d, g in zip(docs, grades) if g == "relevant"]
        if any_irrelevant(grades):
            with stage("Searching the web"):
                web_results = web_search(question)
            context = "\n\n".join(
                part for part in (format_chunks(relevant), format_web(web_results)) if part
            )
        else:
            context = format_chunks(relevant)

    answer, hallucination, h_reason, h_retries, refused = generate_grounded(
        client, context, question
    )

    check_verdict, check_reason, check_rounds = "skipped", "", 0
    if not refused:
        with stage("Checking if answer resolves the question"):
            check_verdict, check_reason = answers_question(question, answer, client)
        while check_verdict == "no" and check_rounds < MAX_ANSWER_ROUNDS:
            check_rounds += 1
            with stage(f"Searching the web (round {check_rounds})"):
                fresh = web_search(question)
            web_results = web_results + fresh
            web_block = format_web(fresh)
            context = f"{context}\n\n{web_block}" if context else web_block
            answer, hallucination, h_reason, h_retries, refused = generate_grounded(
                client, context, question
            )
            if refused:
                check_verdict = "skipped"
                break
            check_verdict, check_reason = answers_question(question, answer, client)

    return {
        "question": question,
        "route": route,
        "docs": docs,
        "grades": grades,
        "web_results": web_results,
        "context": context,
        "answer": answer,
        "hallucination": "refused" if refused else hallucination,
        "hallucination_reason": h_reason,
        "hallucination_retries": h_retries,
        "answers_question": check_verdict,
        "answers_question_reason": check_reason,
        "answers_question_rounds": check_rounds,
    }
