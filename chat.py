import faiss
import pickle
import numpy as np
from ollama import embed, chat

# ----------------------------------------
# Load FAISS Index
# ----------------------------------------

index = faiss.read_index("vector_store/faiss_index.bin")

print("FAISS index loaded.")

# ----------------------------------------
# Load Chunks
# ----------------------------------------

with open("vector_store/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

print(f"Loaded {len(chunks)} chunks.")

# ----------------------------------------
# Ask Question
# ----------------------------------------

question = input("\nAsk a question: ")

# ----------------------------------------
# Generate Question Embedding
# ----------------------------------------

response = embed(
    model="nomic-embed-text",
    input=question
)

question_embedding = response["embeddings"][0]

# Convert to NumPy
question_embedding = np.array(
    [question_embedding],
    dtype="float32"
)

# ----------------------------------------
# Search FAISS
# ----------------------------------------

k = 3

distances, indices = index.search(
    question_embedding,
    k
)

# ----------------------------------------
# Display Results
# ----------------------------------------

print("\nTop Retrieved Chunks\n")

retrieved_chunks = []

for i, chunk_index in enumerate(indices[0]):

    print(f"Rank {i+1}")
    print(f"Chunk Index : {chunk_index}")
    print(f"Distance    : {distances[0][i]:.4f}")
    print(chunks[chunk_index])
    print("-" * 60)

    retrieved_chunks.append(chunks[chunk_index])

# ----------------------------------------
# Build Context
# ----------------------------------------

context = "\n\n".join(retrieved_chunks)

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
            "content":
            "Answer ONLY using the provided context. "
            "If the answer is not in the context, say "
            "'I don't know based on the provided document.'"
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

print("AI:\n")
print(response.message.content)