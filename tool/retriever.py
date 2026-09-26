"""Retrieve Documents stage: embed the question, Top-K cosine search."""

import numpy as np

from config import EMBEDDING_DEPLOYMENT


def cosine_similarity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return float(
        np.dot(vector1, vector2)
        / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    )


def embed_text(client, text):
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text,
    )
    return response.data[0].embedding


def retrieve(question, knowledge_base, client, top_k=3):
    """Return top_k [{text, source, score}] sorted by cosine similarity."""
    question_embedding = embed_text(client, question)
    scored = [
        {
            "text": item["text"],
            "source": item["source"],
            "score": cosine_similarity(question_embedding, item["embedding"]),
        }
        for item in knowledge_base
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
