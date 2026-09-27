from adaptive import answer_question
from config import build_client
from tool.knowledge_base import (
    build_chunks,
    ensure_embeddings,
    load_documents,
)


def print_result(result):
    print(f"\nRoute: {result['route']}\n")

    for round_info in result.get("retrieval_rounds", []):
        print(
            f"Retrieval round {round_info['round']}: {round_info['query']}\n"
            f"Decision: {round_info['decision']} — {round_info['evidence_gap']}\n"
        )

    if result["docs"]:
        print("Top Retrieved Chunks\n")
        for chunk, grade in zip(result["docs"], result["grades"]):
            print(f"Score: {chunk['score']:.4f} | Source: {chunk['source']} | Grade: {grade}")
            print(chunk["text"])
            print("-" * 60)

    if result["web_results"]:
        print("\nWeb Search Results\n")
        for item in result["web_results"]:
            print(f"{item['title']}")
            print(item["url"])
            print("-" * 60)

    print("\nContext Sent To LLM:\n")
    print(result["context"])

    print(
        f"\nHallucination check: {result['hallucination']}"
        f" (retries: {result['hallucination_retries']})"
    )
    if result["hallucination_reason"]:
        print(f"Reason: {result['hallucination_reason']}")

    print(
        f"\nAnswers question: {result['answers_question']}"
        f" (extra web rounds: {result['answers_question_rounds']})"
    )
    if result["answers_question_reason"]:
        print(f"Reason: {result['answers_question_reason']}")

    print("\nAI:\n")
    print(result["answer"])


def run_chat(client=None, knowledge_base=None):
    """Continuous chat loop. Type 'quit' to exit. KB loads once."""
    if client is None:
        client = build_client()
    if knowledge_base is None:
        documents = load_documents()
        chunks = build_chunks(documents)
        knowledge_base = ensure_embeddings(chunks, client)
        print(f"\nStored {len(knowledge_base)} chunks.\n")

    print("Ask questions below. Type 'quit' to exit.\n")
    while True:
        try:
            question = input("Ask a question: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if question.strip().lower() == "quit":
            print("Bye.")
            break
        if not question.strip():
            continue
        try:
            result = answer_question(question, knowledge_base, client)
        except Exception as exc:
            print(f"\nError: {exc} (try another question)\n")
            continue
        print_result(result)


if __name__ == "__main__":
    run_chat()
