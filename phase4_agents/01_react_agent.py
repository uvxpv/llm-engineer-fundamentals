"""
01_react_agent.py — ReAct (Reason + Act) agent with tool use.

Demonstrates:
- Defining tools as JSON schemas and registering Python functions
- The ReAct loop: send messages → LLM decides to call a tool or answer → repeat
- How the LLM drives the loop by emitting tool_call objects in its response
- Multi-step reasoning across multiple tool calls in a single question

The three tools:
  calculator            — evaluates a math expression
  get_weather           — returns fake weather data (no API key needed)
  search_knowledge_base — queries an in-memory vector store (same corpus as Phase 2/3)
"""

import json
import os

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Backend ──────────────────────────────────────────────────────────────────

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

# ── Knowledge base (same corpus as Phase 2 / 3) ──────────────────────────────

_DOCUMENTS = [
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


def _build_knowledge_base():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()
    collection = client.create_collection("docs")

    chunks, ids, metas = [], [], []
    for doc in _DOCUMENTS:
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


print("Building knowledge base...")
_embed_model, _collection = _build_knowledge_base()
print("Knowledge base ready\n")

# ── Tool implementations ──────────────────────────────────────────────────────


def calculator(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression."""
    try:
        # ast.literal_eval won't work for math; restrict to safe characters only.
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only basic arithmetic is supported"
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    """Return fake weather for demonstration purposes."""
    fake_data = {
        "berlin": "Cloudy, 12°C, light wind from the west.",
        "new york": "Sunny, 22°C, humidity 55%.",
        "tokyo": "Rainy, 18°C, heavy showers expected.",
    }
    return fake_data.get(city.lower(), f"No weather data available for '{city}'.")


def search_knowledge_base(query: str) -> str:
    """Search the in-memory knowledge base and return the top 2 matching chunks."""
    q_vec = _embed_model.encode([query], normalize_embeddings=True).tolist()
    results = _collection.query(
        query_embeddings=q_vec, n_results=2, include=["documents", "metadatas"]
    )
    parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        parts.append(f"[{meta['source']}] {doc}")
    return "\n\n".join(parts) if parts else "No relevant results found."


# Map tool names to Python callables.
TOOL_DISPATCH = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base,
}

# ── Tool schemas (OpenAI function-calling format) ────────────────────────────
# These JSON schemas are what gets sent to the LLM so it knows what tools exist
# and what arguments each tool expects.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '12 * 8 + 3'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Berlin'",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the internal knowledge base about Python, Git, and Linux. "
                "Use this to answer technical questions about these topics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]

# ── ReAct loop ────────────────────────────────────────────────────────────────


def run_agent(question: str) -> str:
    """
    Run the ReAct loop for a given question.

    The loop:
      1. Send the current message history (+ tool schemas) to the LLM.
      2. If the LLM returns tool_calls: execute each tool, append results, go to 1.
      3. If the LLM returns plain content: that is the final answer — return it.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with access to tools. "
                "Use the tools when needed to answer precisely. "
                "Always show your reasoning before calling a tool."
            ),
        },
        {"role": "user", "content": question},
    ]

    max_iterations = 10  # safety cap to prevent infinite loops
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"  [iteration {iteration}] calling LLM...")

        response = requests.post(
            CHAT_URL,
            headers=HEADERS,
            json={"model": MODEL, "messages": messages, "tools": TOOLS},
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        finish_reason = choice["finish_reason"]
        message = choice["message"]
        # print how many choices there are
        print(
            "number of choices, finish_reason:",
            len(response.json()["choices"]),
            finish_reason,
        )

        # Append the assistant's response to the history so the LLM has context.
        messages.append(message)

        if finish_reason == "stop":
            # Model signalled it is done — return the final answer.
            return message.get("content", "")
        elif finish_reason == "length":
            # Model hit max_tokens mid-generation — content is truncated.
            return f"[truncated] {message.get('content', '')}"
        elif finish_reason != "tool_calls":
            # Unexpected stop reason — surface it rather than silently looping.
            return f"[unexpected finish_reason={finish_reason!r}] {message.get('content', '')}"
        tool_calls = message.get("tool_calls") or []

        # finish_reason == "tool_calls" → execute tools and loop.

        # Execute every tool the LLM requested and append the results.
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])

            print(f"  [tool call] {fn_name}({fn_args})")
            result = TOOL_DISPATCH[fn_name](**fn_args)
            print(f"  [tool result] {result[:120]}")

            # The tool result goes back as a "tool" role message.
            # The LLM needs the tool_call_id to match result to request.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                }
            )

    return "Agent reached max iterations without a final answer."


# ── Run test questions ────────────────────────────────────────────────────────

QUESTIONS = [
    "What is 347 * 28?",
    "What does git rebase do, and what is 6 * 7?",
]

for question in QUESTIONS:
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)
    answer = run_agent(question)
    print(f"\nFinal answer:\n{answer}\n")
