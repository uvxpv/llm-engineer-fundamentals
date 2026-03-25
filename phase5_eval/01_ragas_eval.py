"""
01_ragas_eval.py — RAG pipeline evaluation with RAGAS.

Demonstrates:
- Building a golden evaluation dataset: (question, ground_truth, answer, retrieved_contexts)
- Three RAGAS metrics (all LLM-graded):
    faithfulness        — is the answer supported by the retrieved context?
    context_precision   — are the top-ranked chunks actually relevant to the question?
    context_recall      — does the retrieved set cover what the ground truth needs?
  (answer_relevancy is skipped: it requires an embedding endpoint — RAGAS 0.4 dropped
   support for local sentence-transformers in its metrics layer)
- Why evaluation needs an LLM: the evaluator IS an LLM call
- One out-of-corpus question to stress-test faithfulness

Same corpus as Phase 2/3 (Python, Git, Linux). Same retriever pattern.
"""

import os

import chromadb
import requests
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.llms import llm_factory
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Backend ──────────────────────────────────────────────────────────────────
# RAGAS 0.4+ uses llm_factory, which takes any OpenAI-compatible client.
# Both OpenRouter and Ollama expose OpenAI-compatible APIs, so we point the
# standard openai.OpenAI client at their base URLs — no actual OpenAI service needed.

BACKEND = os.getenv("BACKEND", "openrouter")

if BACKEND == "openrouter":
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = os.getenv("OPENROUTER_MODEL")
    HEADERS = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    _openai_client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    _async_client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
else:
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    MODEL = os.getenv("OLLAMA_MODEL")
    HEADERS = {"Content-Type": "application/json"}
    _openai_client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
    )
    _async_client = AsyncOpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1",
    )

CHAT_URL = f"{BASE_URL}/chat/completions"

# llm_factory creates RAGAS's native InstructorLLM from any OpenAI-compatible client.
# AsyncOpenAI is needed because RAGAS runs metrics concurrently via asyncio.
ragas_llm = llm_factory(MODEL, client=_async_client)

# ── Corpus + vector store (same as Phase 2) ──────────────────────────────────

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


# ── RAG query (returns answer + retrieved chunks for RAGAS) ──────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the context does not contain enough information, say so explicitly."
)


def rag(question: str, top_k: int = 3) -> tuple[str, list[str]]:
    """Run RAG and return (answer, list_of_retrieved_chunks)."""
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
    answer = response.json()["choices"][0]["message"]["content"]
    return answer, chunks


# ── Golden evaluation dataset ─────────────────────────────────────────────────
# Each row: question + ground_truth (what a correct answer should say).
# We run the RAG pipeline on each question to collect the actual answer and
# the retrieved chunks. RAGAS then grades the pipeline output against both.

EVAL_QUESTIONS = [
    {
        "question": "Who created Python and when was it first released?",
        "ground_truth": "Python was created by Guido van Rossum and first released in 1991.",
    },
    {
        "question": "What does git rebase do?",
        "ground_truth": (
            "git rebase replays commits on top of a different base, producing a linear history."
        ),
    },
    {
        "question": "Which Linux distributions are mentioned in the knowledge base?",
        "ground_truth": "Ubuntu, Fedora, and Arch are mentioned as Linux distributions.",
    },
    {
        "question": "How does Git handle branching internally?",
        "ground_truth": (
            "Branches in Git are lightweight pointers to commits, making branching extremely cheap."
        ),
    },
    {
        # Out-of-corpus: the answer is not in the knowledge base.
        # A faithful system should refuse to answer or say it doesn't know.
        # A hallucinating system will answer anyway — faithfulness score drops.
        "question": "What is the capital of France?",
        "ground_truth": "The capital of France is Paris.",
    },
]

print("Running RAG pipeline on evaluation questions...")
samples = []
for row in EVAL_QUESTIONS:
    print(f"  Q: {row['question']}")
    answer, contexts = rag(row["question"])
    samples.append(
        SingleTurnSample(
            user_input=row["question"],
            response=answer,
            retrieved_contexts=contexts,
            reference=row["ground_truth"],
        )
    )

dataset = EvaluationDataset(samples=samples)
print(f"\nBuilt evaluation dataset with {len(samples)} samples\n")

# ── Evaluate ──────────────────────────────────────────────────────────────────
# RAGAS calls the judge LLM internally — each metric runs its own LLM prompts.
metrics = [
    Faithfulness(llm=ragas_llm),
    ContextPrecision(llm=ragas_llm),
    ContextRecall(llm=ragas_llm),
]

# Increase timeout for local models (Ollama runs inference sequentially, so
# concurrent metric calls queue up). max_workers=1 avoids overloading the server.
run_config = RunConfig(timeout=300, max_workers=1)

print("Evaluating with RAGAS (this makes multiple LLM calls)...")
results = evaluate(dataset=dataset, metrics=metrics, llm=ragas_llm, run_config=run_config)

# ── Print score table ─────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RAGAS Evaluation Results")
print("=" * 60)
scores = results.scores
for i, (row, score) in enumerate(zip(EVAL_QUESTIONS, scores)):
    print(f"\n[{i + 1}] {row['question']}")
    for metric_name, value in score.items():
        v = value if value is not None and value == value else 0  # handle NaN
        bar = "#" * int(v * 20)
        print(f"  {metric_name:<25} {v:.3f}  |{bar}")

print("\n--- Averages ---")
df = results.to_pandas()
for col in df.columns:
    if col not in ("user_input", "response", "retrieved_contexts", "reference"):
        print(f"  {col:<25} {df[col].mean():.3f}")
