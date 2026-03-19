"""
01_chunking.py — Document loading and text chunking strategies.

Demonstrates:
- Representing a small document corpus as plain strings
- Fixed-size chunking with overlap (the most common strategy)
- Sentence-based chunking (split on sentence boundaries)
- Why chunk size and overlap matter for retrieval quality
"""

# ── Corpus ────────────────────────────────────────────────────────────────────
# A small inline corpus — three short documents on different topics.

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


# ── Fixed-size chunking ───────────────────────────────────────────────────────

def chunk_fixed(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    Split text into chunks of `chunk_size` characters with `overlap` chars
    carried over from the previous chunk.

    Overlap ensures that information at chunk boundaries is not lost —
    a sentence split across two chunks will appear in both.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ── Sentence-based chunking ───────────────────────────────────────────────────

def chunk_sentences(text: str, sentences_per_chunk: int = 2) -> list[str]:
    """
    Split on sentence boundaries ('. ') and group N sentences per chunk.

    Keeps semantic units intact — no sentence is split mid-way.
    Less predictable chunk sizes than fixed-size, but better semantic coherence.
    """
    sentences = [s.strip() + "." for s in text.split(". ") if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks


# ── Demo ──────────────────────────────────────────────────────────────────────

for doc in DOCUMENTS:
    print(f"\n{'=' * 60}")
    print(f"Document: {doc['id']}  ({len(doc['text'])} chars)")
    print(f"{'=' * 60}")

    fixed = chunk_fixed(doc["text"], chunk_size=200, overlap=40)
    print(f"\n[Fixed-size] chunk_size=200, overlap=40 → {len(fixed)} chunks")
    for i, chunk in enumerate(fixed):
        print(f"  [{i}] ({len(chunk)} chars) {chunk[:80]}...")

    sents = chunk_sentences(doc["text"], sentences_per_chunk=2)
    print(f"\n[Sentence]  sentences_per_chunk=2 → {len(sents)} chunks")
    for i, chunk in enumerate(sents):
        print(f"  [{i}] {chunk[:80]}...")
