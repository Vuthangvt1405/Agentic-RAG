from ollama import embed, chat
import numpy as np


# ----------------------------------------
# Cosine Similarity Function
# ----------------------------------------
def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# ----------------------------------------
# Read Document
# ----------------------------------------
with open("data/document.txt", "r", encoding="utf-8") as file:
    document = file.read()


# ----------------------------------------
# Split into Chunks
# ----------------------------------------
chunks = [
    sentence.strip()
    for sentence in document.split(".")
    if sentence.strip()
]


# ----------------------------------------
# Generate Embeddings
# ----------------------------------------
knowledge_base = []

for chunk in chunks:

    response = embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    knowledge_base.append({
        "text": chunk,
        "embedding": embedding
    })

print(f"\nStored {len(knowledge_base)} chunks.\n")


# ----------------------------------------
# Ask Question
# ----------------------------------------
question = input("Ask a question: ")


# ----------------------------------------
# Embed Question
# ----------------------------------------
response = embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]


# ----------------------------------------
# Calculate Similarities
# ----------------------------------------
scores = []

for item in knowledge_base:

    score = cosine_similarity(
        question_embedding,
        item["embedding"]
    )

    scores.append({
        "text": item["text"],
        "score": score
    })


# ----------------------------------------
# Sort Scores
# ----------------------------------------
scores.sort(
    key=lambda x: x["score"],
    reverse=True
)


# ----------------------------------------
# Top K Retrieval
# ----------------------------------------
TOP_K = 3

top_chunks = scores[:TOP_K]

print("\nTop Retrieved Chunks\n")

for chunk in top_chunks:
    print(f"Score: {chunk['score']:.4f}")
    print(chunk["text"])
    print("-" * 60)


# ----------------------------------------
# Build Context
# ----------------------------------------
context = "\n\n".join(
    chunk["text"] for chunk in top_chunks
)

print("\nContext Sent To LLM:\n")
print(context)


# ----------------------------------------
# Ask Qwen
# ----------------------------------------
print("\nGenerating Final Answer...\n")

response = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "system",
            "content": (
                "Answer ONLY using the provided context. "
                "If the answer cannot be found in the context, "
                "say 'I don't know based on the provided document.'"
            )
        },
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}
"""
        }
    ]
)


# ----------------------------------------
# Final Answer
# ----------------------------------------
print("AI:\n")
print(response.message.content)