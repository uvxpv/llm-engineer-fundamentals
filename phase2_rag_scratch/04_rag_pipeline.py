"""
04_rag_pipeline.py — Full RAG pipeline without any frameworks.

Demonstrates:
- Wiring chunking → embedding → vector store → retrieval → LLM into one pipeline
- Constructing a retrieval-augmented prompt (context + question)
- How grounding the LLM in retrieved text reduces hallucination
- Asking two questions: one answerable from the corpus, one outside it
"""

import os

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Backend (same pattern as phase 1) ────────────────────────────────────────

BACKEND = os.getenv("BACKEND", "openrouter")

if BACKEND == "openrouter":
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = os.getenv("OPENROUTER_MODEL")
    HEADERS = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
else:
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    MODEL = os.getenv("OLLAMA_MODEL")
    HEADERS = {"Content-Type": "application/json"}

CHAT_URL = f"{BASE_URL}/chat/completions"


def llm(system: str, user: str) -> str:
    response = requests.post(
        CHAT_URL,
        headers=HEADERS,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── Corpus ────────────────────────────────────────────────────────────────────

DOCUMENTS = [
    {
        "id": "python_basics",
        "text": (
            "Python is a high-level, interpreted programming language known for its clear syntax. "
            "It supports multiple programming paradigms including procedural, object-oriented, and functional styles. "
            "Python uses indentation to define code blocks instead of curly braces. "
            "The Python Package Index (PyPI) hosts hundreds of thousands of third-party packages. "
            "Python is widely used in data science, machine learning, web development, and automation. "
            "The language was created by Guido van Rossum and first released in 1991. "
            "Python 3 is the current major version and is not fully backward-compatible with Python 2."
        ),
    },
    {
        "id": "git_basics",
        "text": (
            "Git is a distributed version control system created by Linus Torvalds in 2005. "
            "Every Git repository contains the full history of all changes, not just the latest snapshot. "
            "Branches in Git are lightweight pointers to commits, making branching extremely cheap. "
            "The staging area (index) lets you craft commits precisely before recording them. "
            "git commit records a snapshot of the staged changes with a message describing the intent. "
            "git merge integrates changes from one branch into another. "
            "git rebase replays commits on top of a different base, producing a linear history."
        ),
    },
    {
        "id": "linux_basics",
        "text": (
            "Linux is an open-source operating system kernel first released by Linus Torvalds in 1991. "
            "Distributions like Ubuntu, Fedora, and Arch package the kernel with userland tools. "
            "The file system hierarchy starts at root (/), with standard directories like /etc, /var, and /home. "
            "Processes in Linux are managed with commands like ps, top, kill, and systemctl. "
            "File permissions are represented as read (r), write (w), and execute (x) for owner, group, and others. "
            "The shell (bash, zsh) is the primary interface for interacting with the system. "
            "Package managers like apt and dnf handle software installation and dependency resolution."
        ),
    },
]


def chunk_sentences(text: str, sentences_per_chunk: int = 2) -> list[str]:
    sentences = [s.strip() + "." for s in text.split(". ") if s.strip()]
    return [
        " ".join(sentences[i : i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]


# ── Build vector store ────────────────────────────────────────────────────────

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.EphemeralClient()
collection = client.create_collection("docs")

all_chunks, all_ids, all_metadata = [], [], []
for doc in DOCUMENTS:
    for i, chunk in enumerate(chunk_sentences(doc["text"])):
        all_chunks.append(chunk)
        all_ids.append(f"{doc['id']}_{i}")
        all_metadata.append({"source": doc["id"]})

collection.add(
    ids=all_ids,
    embeddings=embed_model.encode(all_chunks, normalize_embeddings=True).tolist(),
    documents=all_chunks,
    metadatas=all_metadata,
)

print(f"Vector store ready: {len(all_chunks)} chunks\n")


# ── RAG query function ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the information "
    "in the provided context. If the context does not contain enough information "
    "to answer, say so explicitly."
)


def rag(question: str, top_k: int = 3) -> str:
    # 1. Embed the question
    q_embedding = embed_model.encode([question], normalize_embeddings=True).tolist()

    # 2. Retrieve top-k chunks
    results = collection.query(
        query_embeddings=q_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]

    # 3. Build augmented prompt
    context = "\n\n".join(f"[{src}]\n{chunk}" for src, chunk in zip(sources, chunks))
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    # 4. Call LLM
    return llm(SYSTEM_PROMPT, user_message)


# ── Run two questions ─────────────────────────────────────────────────────────

questions = [
    "Who created Git and why was it built?",
    "What is the capital of France?",  # outside the corpus — model should say so
]

for question in questions:
    print(f"Question: {question}")
    answer = rag(question)
    print(f"Answer:   {answer}\n")
    print("-" * 60 + "\n")
