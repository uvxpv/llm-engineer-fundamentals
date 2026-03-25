"""
02_llm_judge.py — LLM-as-judge evaluation pattern.

Demonstrates:
- The dominant 2024-2025 production eval pattern (used by OpenAI Evals, MT-Bench, Prometheus)
- Scoring RAG answers on three criteria without needing a ground truth:
    groundedness   — is every claim in the answer supported by the retrieved context?
    relevance      — does the answer actually address what the user asked?
    completeness   — does the answer cover the key points the context can support?
- Structured output: judge returns JSON {"score": 1-5, "reason": "..."} per criterion
- Why this scales: works on open-ended tasks where metric-based frameworks can't

No framework needed — just an LLM and a well-designed rubric prompt.
Same corpus and eval questions as 01_ragas_eval.py.
"""

import json
import os

import chromadb
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Backend (same pattern as all prior phases) ────────────────────────────────

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

# ── Corpus + vector store ─────────────────────────────────────────────────────

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


def _build_vector_store():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()
    collection = client.create_collection("docs")
    chunks, ids, metas = [], [], []
    for doc in DOCUMENTS:
        sentences = [s.strip() + "." for s in doc["text"].split(". ") if s.strip()]
        for i in range(0, len(sentences), 2):
            chunk = " ".join(sentences[i : i + 2])
            chunks.append(chunk)
            ids.append(f"{doc['id']}_{i}")
            metas.append({"source": doc["id"]})
    collection.add(
        ids=ids,
        embeddings=embed_model.encode(chunks, normalize_embeddings=True).tolist(),
        documents=chunks,
        metadatas=metas,
    )
    return embed_model, collection


print("Building vector store...")
_embed_model, _collection = _build_vector_store()
print("Vector store ready\n")

# ── RAG pipeline ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the context does not contain enough information, say so explicitly."
)


def rag(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    q_vec = _embed_model.encode([question], normalize_embeddings=True).tolist()
    results = _collection.query(
        query_embeddings=q_vec, n_results=top_k, include=["documents", "metadatas"]
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    context = "\n\n".join(f"[{src}]\n{chunk}" for src, chunk in zip(sources, chunks))
    response = requests.post(
        CHAT_URL,
        headers=HEADERS,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"], chunks


# ── Pydantic schema for structured judge output ───────────────────────────────
# Same pattern as phase1/03_structured_output.py.
# The judge LLM must return JSON; we validate it with Pydantic.


class CriterionScore(BaseModel):
    score: int  # 1–5
    reason: str


class JudgeOutput(BaseModel):
    groundedness: CriterionScore
    relevance: CriterionScore
    completeness: CriterionScore


JUDGE_SCHEMA = json.dumps(
    {
        "groundedness": {"score": 4, "reason": "All claims are directly supported by the context."},
        "relevance": {"score": 5, "reason": "The answer directly addresses the question."},
        "completeness": {"score": 3, "reason": "Covers the main point but misses one detail."},
    },
    indent=2,
)

JUDGE_SYSTEM = (
    "You are an expert evaluator of question-answering systems. "
    "You will be given a question, retrieved context, and an answer. "
    "Score the answer on three criteria using a 1-5 scale:\n\n"
    "  groundedness  — every claim in the answer is supported by the context (1=hallucinated, 5=fully grounded)\n"
    "  relevance     — the answer addresses what the question asked (1=off-topic, 5=directly on-topic)\n"
    "  completeness  — the answer covers the key information the context can support (1=major gaps, 5=thorough)\n\n"
    "Respond ONLY with a JSON object in exactly this format:\n\n"
    f"{JUDGE_SCHEMA}"
)


def judge(question: str, context_chunks: list[str], answer: str) -> JudgeOutput | None:
    context = "\n\n".join(context_chunks)
    user_msg = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    response = requests.post(
        CHAT_URL,
        headers=HEADERS,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    try:
        return JudgeOutput.model_validate_json(raw)
    except ValidationError as e:
        print(f"  [parse error] {e}")
        print(f"  [raw output] {raw[:300]}")
        return None


# ── Evaluation questions (same as 01_ragas_eval.py, no ground truth needed) ───

QUESTIONS = [
    "Who created Python and when was it first released?",
    "What does git rebase do?",
    "Which Linux distributions are mentioned in the knowledge base?",
    "How does Git handle branching internally?",
    "What is the capital of France?",  # out-of-corpus — tests groundedness
]

# ── Run evaluation ────────────────────────────────────────────────────────────

print("Running LLM-as-judge evaluation...\n")

all_scores: list[JudgeOutput] = []
results_log = []

for question in QUESTIONS:
    print(f"Q: {question}")
    answer, contexts = rag(question)
    print(f"  Answer: {answer[:120].replace(chr(10), ' ')}...")
    judgment = judge(question, contexts, answer)
    if judgment:
        all_scores.append(judgment)
        results_log.append((question, answer, judgment))
        g, r, c = judgment.groundedness.score, judgment.relevance.score, judgment.completeness.score
        print(f"  Groundedness: {g}/5  Relevance: {r}/5  Completeness: {c}/5")
    print()

# ── Summary table ─────────────────────────────────────────────────────────────

print("=" * 60)
print("LLM-as-Judge Summary")
print("=" * 60)
print(f"{'Question':<45} {'Grd':>4} {'Rel':>4} {'Cmp':>4}")
print("-" * 60)
for question, _, j in results_log:
    short = question[:43] + ".." if len(question) > 43 else question
    print(f"{short:<45} {j.groundedness.score:>4} {j.relevance.score:>4} {j.completeness.score:>4}")

if all_scores:
    print("-" * 60)
    avg_g = sum(j.groundedness.score for j in all_scores) / len(all_scores)
    avg_r = sum(j.relevance.score for j in all_scores) / len(all_scores)
    avg_c = sum(j.completeness.score for j in all_scores) / len(all_scores)
    print(f"{'Average':<45} {avg_g:>4.1f} {avg_r:>4.1f} {avg_c:>4.1f}")

print("\n--- Reasons ---")
for question, _, j in results_log:
    print(f"\nQ: {question}")
    print(f"  Groundedness ({j.groundedness.score}): {j.groundedness.reason}")
    print(f"  Relevance    ({j.relevance.score}): {j.relevance.reason}")
    print(f"  Completeness ({j.completeness.score}): {j.completeness.reason}")
