"""Adaptive RAG pipeline (see asset/workflow.png), plain Python.

Question -> Routing -> vectorstore: Retrieve -> Grade ->
    all relevant: Generate Answer (chunks only)
    any irrelevant: Web Search -> Generate Answer (relevant chunks + web)
  -> web_search route: Web Search -> Generate Answer (web only)
"""

from config import LLM_DEPLOYMENT
from router import dispatch
from tool.answer_check import answers_question
from tool.grader import grade_docs
from tool.hallucination import check_hallucination
from tool.loading import stage
from tool.query_rewrite import rewrite_query
from tool.retrieval_decision import decide_next_retrieval
from tool.retriever import retrieve
from tool.web_search import web_search

REFUSAL = "I don't know based on the provided documents."
MAX_HALLUCINATION_RETRIES = 3
MAX_ANSWER_ROUNDS = 2
MAX_LOCAL_RETRIEVAL_ROUNDS = 2


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


def _doc_key(doc):
    return doc["source"], " ".join(doc["text"].lower().split())


def answer_question(question, knowledge_base, client, top_k=3):
    """Run the adaptive RAG pipeline with at most two local retrieval rounds."""
    with stage("Routing question"):
        route = dispatch(question)["route"]

    docs, grades, web_results, retrieval_rounds = [], [], [], []

    if route == "web_search":
        with stage("Searching the web"):
            web_results = web_search(question)
        context = format_web(web_results)
    else:
        previous_queries, seen_docs, relevant = [], set(), []
        evidence_gap = ""
        action = "retrieve_again"

        for round_number in range(1, MAX_LOCAL_RETRIEVAL_ROUNDS + 1):
            with stage(f"Rewriting retrieval query (round {round_number})"):
                query, rewrite_reason = rewrite_query(
                    question, previous_queries, evidence_gap, round_number, client
                )
            previous_queries.append(query)

            with stage(f"Retrieving documents (round {round_number})"):
                candidates = retrieve(query, knowledge_base, client, top_k=top_k)
            round_docs = [doc for doc in candidates if _doc_key(doc) not in seen_docs]
            seen_docs.update(_doc_key(doc) for doc in round_docs)

            with stage(f"Grading documents (round {round_number})"):
                round_grades = grade_docs(question, round_docs, client) if round_docs else []
            round_relevant = [
                doc for doc, grade in zip(round_docs, round_grades) if grade == "relevant"
            ]
            docs.extend(round_docs)
            grades.extend(round_grades)
            relevant.extend(round_relevant)

            with stage(f"Deciding next retrieval step (round {round_number})"):
                action, evidence_gap = decide_next_retrieval(
                    question, relevant, round_docs, round_grades, round_number, client
                )
            retrieval_rounds.append(
                {
                    "round": round_number,
                    "query": query,
                    "rewrite_reason": rewrite_reason,
                    "docs": round_docs,
                    "grades": round_grades,
                    "decision": action,
                    "evidence_gap": evidence_gap,
                }
            )

            if action != "retrieve_again":
                break
            if round_number == MAX_LOCAL_RETRIEVAL_ROUNDS:
                action = "web_search"
                retrieval_rounds[-1]["decision"] = action
                retrieval_rounds[-1]["evidence_gap"] = (
                    "Maximum local retrieval rounds reached. " + evidence_gap
                )

        if action == "web_search":
            with stage("Searching the web"):
                web_results = web_search(evidence_gap or question)
        context = "\n\n".join(
            part for part in (format_chunks(relevant), format_web(web_results)) if part
        )

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
        "retrieval_rounds": retrieval_rounds,
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
