"""
03_vector_store.py — Store and query vectors with ChromaDB.

Demonstrates:
- Creating an ephemeral (in-memory) ChromaDB collection
- Chunking documents, embedding each chunk, upserting into the collection
- Querying the collection with a natural-language question
- Reading back retrieved chunks and their similarity distances
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── Corpus (same as 01_chunking.py) ───────────────────────────────────────────

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


# ── Setup ─────────────────────────────────────────────────────────────────────

model = SentenceTransformer("all-MiniLM-L6-v2")

# EphemeralClient keeps everything in memory — nothing written to disk.
client = chromadb.EphemeralClient()
collection = client.create_collection("docs")

# ── Ingest ────────────────────────────────────────────────────────────────────

all_chunks = []
all_ids = []
all_metadata = []

for doc in DOCUMENTS:
    chunks = chunk_sentences(doc["text"], sentences_per_chunk=2)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{doc['id']}_{i}")
        all_metadata.append({"source": doc["id"]})

embeddings = model.encode(all_chunks, normalize_embeddings=True).tolist()

collection.add(
    ids=all_ids,
    embeddings=embeddings,
    documents=all_chunks,
    metadatas=all_metadata,
)

print(f"Ingested {len(all_chunks)} chunks from {len(DOCUMENTS)} documents\n")

# ── Query ─────────────────────────────────────────────────────────────────────

QUERIES = [
    "Who created Python and when?",
    "How does Git branching work?",
    "How do I manage file permissions in Linux?",
]

for query in QUERIES:
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2,
        include=["documents", "metadatas", "distances"],
    )

    print(f"Query: {query}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        # ChromaDB returns L2 distance by default; lower = more similar.
        print(f"  [{i+1}] source={meta['source']}  distance={dist:.4f}")
        print(f"       {doc[:100]}...")
    print()
