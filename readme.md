# 🧠 Manual RAG Chatbot with FAISS & Ollama

A Retrieval-Augmented Generation (RAG) chatbot built completely from scratch using **Ollama**, **nomic-embed-text**, **FAISS**, and **Qwen 2.5** without relying on LangChain or other RAG frameworks.

This project demonstrates the complete RAG pipeline, including document chunking, embedding generation, vector search, Top-K retrieval, and answer generation using a local Large Language Model (LLM).

---

## 🚀 Features

- Document chunking
- Embedding generation using `nomic-embed-text`
- Vector similarity search using FAISS
- Top-K retrieval
- Local LLM inference with Qwen 2.5
- Context-aware question answering
- Fully offline (no OpenAI API required)

---

# 🏗 Project Structure

```
manual-rag-faiss/
│
├── data/
│   └── document.txt
│
├── vector_store/
│   ├── faiss_index.bin
│   └── chunks.pkl
│
├── app.py
├── ingest.py
├── chat.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📖 How It Works

### Step 1 — Document Ingestion

The document is read and split into smaller chunks.

```
Document
      │
      ▼
 Chunking
```

---

### Step 2 — Generate Embeddings

Each chunk is converted into a 768-dimensional embedding using:

- `nomic-embed-text`

```
Chunk 1 ──► Embedding
Chunk 2 ──► Embedding
Chunk 3 ──► Embedding
...
```

---

### Step 3 — Store in FAISS

All embeddings are stored inside a FAISS index for efficient similarity search.

```
Embeddings
      │
      ▼
 FAISS Index
```

---

### Step 4 — User Question

When a user asks a question:

```
What's the withdrawal limit?
```

The same embedding model converts the question into a vector.

```
Question
     │
     ▼
Embedding
```

---

### Step 5 — Similarity Search

FAISS compares the question embedding against all stored vectors and retrieves the most relevant chunks using Top-K retrieval.

```
Question Vector
        │
        ▼
     FAISS Search
        │
        ▼
 Top 3 Relevant Chunks
```

---

### Step 6 — Answer Generation

The retrieved chunks are sent as context to the local Qwen 2.5 model.

```
Retrieved Chunks
        │
        ▼
     Qwen 2.5
        │
        ▼
 Final Answer
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/manual-rag-faiss.git
cd manual-rag-faiss
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Build the Vector Index

Run

```bash
python ingest.py
```

This will:

- Read the document
- Generate embeddings
- Build the FAISS index
- Save the vector store

---

# 💬 Start the Chatbot

```bash
python chat.py
```

Example

```
Ask a question:

What's the withdrawal limit?
```

Output

```
The withdrawal limit for the savings account is 50,000 PKR per day.
```

---

# 🛠 Technologies Used

- Python
- Ollama
- nomic-embed-text
- FAISS
- NumPy
- Qwen 2.5

---

# 🎯 Future Improvements

- Multi-document ingestion
- Incremental indexing
- Metadata filtering
- Hybrid Search (BM25 + FAISS)
- Re-ranking
- LangChain integration
- AI Agents
- MCP (Model Context Protocol)
- FastAPI backend
- Web interface
- Docker deployment

---

# 📚 Learning Objectives

This project was built to understand the internal mechanics of Retrieval-Augmented Generation without using high-level frameworks.

The focus was to learn:

- Document chunking
- Embeddings
- Cosine similarity
- Vector databases
- FAISS indexing
- Top-K retrieval
- Context construction
- Local LLM inference

before moving on to production-grade RAG systems.

---

# ⭐ Acknowledgements

Built as part of a hands-on journey toward becoming an AI Engineer by implementing every stage of a RAG pipeline from scratch.
