"""
02_rag_chain.py — RAG pipeline rebuilt with LangChain.

Demonstrates:
- RecursiveCharacterTextSplitter instead of hand-rolled chunking
- HuggingFaceEmbeddings wrapping sentence-transformers
- Chroma vector store via LangChain's integration
- LCEL retrieval chain: retriever → prompt → model → parser
- Contrast with Phase 2's manual pipeline — same result, far less plumbing
"""

import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Model setup ──────────────────────────────────────────────────────────────

BACKEND = os.getenv("BACKEND", "openrouter")

if BACKEND == "openrouter":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("OPENROUTER_MODEL"),
    )
else:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL"),
    )

# ── Corpus (same documents as Phase 2) ──────────────────────────────────────

RAW_DOCS = [
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

# ── 1. Chunking with RecursiveCharacterTextSplitter ─────────────────────────
# LangChain's splitter tries multiple separators (paragraphs → sentences → words)
# in order, picking the first that produces chunks within the size limit.
# This replaces our hand-rolled chunk_sentences() from Phase 2.

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", ". ", " ", ""],
)

documents = []
for raw in RAW_DOCS:
    chunks = splitter.create_documents(
        texts=[raw["text"]],
        metadatas=[{"source": raw["id"]}],
    )
    documents.extend(chunks)

print(f"Split {len(RAW_DOCS)} documents into {len(documents)} chunks\n")

# ── 2. Embeddings + Vector store ────────────────────────────────────────────
# LangChain wraps sentence-transformers and ChromaDB so you don't wire them manually.

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Vector store ready\n")

# ── 3. RAG chain with LCEL ─────────────────────────────────────────────────
# The chain: retrieve context → format prompt → call LLM → parse output.

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question using ONLY the provided context. "
    "If the context does not contain enough information, say so explicitly.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 4. Run the same two questions from Phase 2 ─────────────────────────────

questions = [
    "Who created Git and why was it built?",
    "What is the capital of France?",  # outside the corpus
]

for question in questions:
    print(f"Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"Answer:   {answer}\n")
    print("-" * 60 + "\n")
