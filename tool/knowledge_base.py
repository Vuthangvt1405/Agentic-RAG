"""Document loading, chunking, and cached embeddings (shared by app.py and tools)."""

import hashlib
import os
import pickle

from config import EMBEDDING_DEPLOYMENT

DATA_DIR = "data"
CACHE_PATH = os.path.join("vector_store", "embeddings_cache.pkl")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "100"))


def read_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages_text = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            print(f"Warning: could not read page {page_number} of {path}: {exc}")
            continue
        if text.strip():
            pages_text.append(text)
        else:
            print(f"Warning: no text on page {page_number} of {path} (scanned image?).")
    return "\n".join(pages_text)


def load_documents(data_dir=DATA_DIR):
    """Scan a directory for .txt/.pdf files. Returns [(source, text)]."""
    documents = []
    for filename in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, filename)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".txt":
            documents.append((filename, read_txt(path)))
        elif ext == ".pdf":
            documents.append((filename, read_pdf(path)))
        else:
            print(f"Warning: skipping unsupported file {filename}")
    documents = [(source, text) for source, text in documents if text.strip()]
    if not documents:
        raise SystemExit(f"No readable documents found in {data_dir}/ (.txt, .pdf).")
    for source, text in documents:
        print(f"Loaded {source} ({len(text)} chars).")
    return documents


def build_chunks(documents):
    """Split documents into sentence chunks tagged with source."""
    chunks = []
    for source, text in documents:
        for sentence in text.split("."):
            sentence = sentence.strip()
            if sentence:
                chunks.append({"text": sentence, "source": source})
    return chunks


def chunk_key(text, source):
    normalized = f"{EMBEDDING_DEPLOYMENT}\n{source}\n{text.strip()}".encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def load_cache(cache_path=CACHE_PATH):
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as file:
            return pickle.load(file)
    return {}


def save_cache(cache, cache_path=CACHE_PATH):
    parent = os.path.dirname(cache_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(cache_path, "wb") as file:
        pickle.dump(cache, file)


def embed_batch(client, texts):
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in response.data]


def ensure_embeddings(chunks, client, cache_path=CACHE_PATH, batch_size=EMBED_BATCH_SIZE):
    """Return [{text, source, embedding}], embedding only cache misses in batches."""
    cache = load_cache(cache_path)
    keys = [chunk_key(chunk["text"], chunk["source"]) for chunk in chunks]
    missing = [(i, key) for i, key in enumerate(keys) if key not in cache]

    if missing:
        print(f"Cache: {len(keys) - len(missing)} hits, {len(missing)} to embed.\n")
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            texts = [chunks[i]["text"] for i, _ in batch]
            embeddings = embed_batch(client, texts)
            for (i, key), embedding in zip(batch, embeddings):
                cache[key] = embedding
            print(f"Embedded {min(start + batch_size, len(missing))}/{len(missing)}.")
        save_cache(cache, cache_path)
        print("Cache saved.")
    else:
        print(f"Cache: all {len(keys)} embeddings reused, no API calls.\n")

    return [
        {"text": chunk["text"], "source": chunk["source"], "embedding": cache[key]}
        for chunk, key in zip(chunks, keys)
    ]
