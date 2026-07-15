import faiss
import pickle
import numpy as np
from ollama import embed

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

print(f"\nFound {len(chunks)} chunks.\n")

# ----------------------------------------
# Generate Embeddings
# ----------------------------------------

embeddings = []

for index, chunk in enumerate(chunks, start=1):

    response = embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]

    embeddings.append(embedding)

    print(f"Generated embedding for Chunk {index}")

# ----------------------------------------
# Convert to NumPy Array
# ----------------------------------------

embeddings = np.array(embeddings).astype("float32")

print("\nEmbeddings Shape:", embeddings.shape)

# ----------------------------------------
# Create FAISS Index
# ----------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

print(f"\nCreated FAISS Index with dimension {dimension}")

# ----------------------------------------
# Add Embeddings to Index
# ----------------------------------------

index.add(embeddings)

print(f"Added {index.ntotal} vectors to FAISS.")

# ----------------------------------------
# Save FAISS Index
# ----------------------------------------

faiss.write_index(index, "vector_store/faiss_index.bin")

print("\nFAISS index saved.")

# ----------------------------------------
# Save Chunks
# ----------------------------------------

with open("vector_store/chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("Chunks saved.")

print("\nIngestion Complete!")