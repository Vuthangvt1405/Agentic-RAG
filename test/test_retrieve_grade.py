import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import build_client
from tool.grader import any_irrelevant, grade_docs
from tool.knowledge_base import ensure_embeddings
from tool.retriever import retrieve

# Tiny synthetic corpus: no data/ scan, no real cache touched.
CHUNKS = [
    {"text": "Zero trust network access verifies every request before granting access", "source": "ztna.pdf"},
    {"text": "Micro-segmentation divides the network to limit lateral movement", "source": "ztna.pdf"},
    {"text": "Sourdough bread needs flour, water, salt and a starter culture", "source": "baking.pdf"},
]

client = build_client()
cache_file = os.path.join(tempfile.gettempdir(), "agentic_rag_test_cache.pkl")
if os.path.isfile(cache_file):
    os.remove(cache_file)

kb = ensure_embeddings(CHUNKS, client, cache_path=cache_file)
print(f"KB built: {len(kb)} chunks.")

# --- Retrieve: in-scope question should rank ZTNA chunks first ---
top = retrieve("What is zero trust network access?", kb, client, top_k=3)
for item in top:
    print(f"{item['score']:.4f} | {item['source']} | {item['text'][:60]}")
assert top[0]["source"] == "ztna.pdf", "retrieval ranked off-topic chunk first"
print("retrieve ok: on-topic chunk ranked first.")

# Cache file must now exist; second build reuses it with no API calls.
assert os.path.isfile(cache_file)
kb2 = ensure_embeddings(CHUNKS, client, cache_path=cache_file)

# --- Grade: ZTNA docs relevant, baking doc irrelevant ---
grades = grade_docs("What is zero trust network access?", top, client)
print("grades:", grades)
assert grades[top.index(next(d for d in top if d["source"] == "baking.pdf"))] == "irrelevant"
assert any_irrelevant(grades) is True
print("grade ok: off-topic chunk marked irrelevant, fallback triggered.")

relevant_only = [d for d, g in zip(top, grades) if g == "relevant"]
grades2 = grade_docs("What is zero trust network access?", relevant_only, client)
assert any_irrelevant(grades2) is False
print("grade ok: on-topic chunks all relevant.")

os.remove(cache_file)
print("ALL RETRIEVE+GRADE CHECKS PASSED")
