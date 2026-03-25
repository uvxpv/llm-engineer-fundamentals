"""
02_agent_memory.py — Multi-turn agent with persistent conversation memory.

Demonstrates:
- A REPL loop where the same messages list survives across turns
- That LLM "memory" is just the message history re-sent on every call
- /clear resetting the list — the agent forgets everything instantly
- /history printing the raw message list — making the "context window = memory" concrete

Same tools as 01_react_agent.py (calculator, get_weather, search_knowledge_base).
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

# ── Knowledge base ────────────────────────────────────────────────────────────

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

# ── Tools ─────────────────────────────────────────────────────────────────────


def calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "Error: only basic arithmetic is supported"
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


def get_weather(city: str) -> str:
    fake_data = {
        "berlin": "Cloudy, 12°C, light wind from the west.",
        "new york": "Sunny, 22°C, humidity 55%.",
        "tokyo": "Rainy, 18°C, heavy showers expected.",
    }
    return fake_data.get(city.lower(), f"No weather data available for '{city}'.")


def search_knowledge_base(query: str) -> str:
    q_vec = _embed_model.encode([query], normalize_embeddings=True).tolist()
    results = _collection.query(
        query_embeddings=q_vec, n_results=2, include=["documents", "metadatas"]
    )
    parts = [
        f"[{m['source']}] {d}"
        for d, m in zip(results["documents"][0], results["metadatas"][0])
    ]
    return "\n\n".join(parts) if parts else "No relevant results found."


TOOL_DISPATCH = {
    "calculator": calculator,
    "get_weather": get_weather,
    "search_knowledge_base": search_knowledge_base,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '12 * 8 + 3'"}
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
            "description": "Search the internal knowledge base about Python, Git, and Linux.",
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

# ── Agent step (one LLM call + tool execution if needed) ─────────────────────


def agent_step(messages: list) -> str:
    """
    Run the ReAct loop for the current messages list.
    Mutates messages in-place by appending assistant and tool messages.
    Returns the final answer string.
    """
    max_iterations = 10
    for iteration in range(1, max_iterations + 1):
        response = requests.post(
            CHAT_URL,
            headers=HEADERS,
            json={"model": MODEL, "messages": messages, "tools": TOOLS},
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        finish_reason = choice["finish_reason"]
        message = choice["message"]

        messages.append(message)

        if finish_reason == "stop":
            return message.get("content", "")
        elif finish_reason == "length":
            return f"[truncated] {message.get('content', '')}"
        elif finish_reason != "tool_calls":
            return f"[unexpected finish_reason={finish_reason!r}] {message.get('content', '')}"

        # finish_reason == "tool_calls" → run tools, loop again
        tool_calls = message.get("tool_calls") or []
        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            print(f"  [tool call] {fn_name}({fn_args})")
            result = TOOL_DISPATCH[fn_name](**fn_args)
            print(f"  [tool result] {result[:120]}")
            messages.append(
                {"role": "tool", "tool_call_id": tc["id"], "content": result}
            )

    return "Agent reached max iterations without a final answer."


# ── REPL ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "You remember everything said earlier in this conversation. "
    "Use tools when needed. Be concise."
)

# The messages list is the entire memory of the agent.
# It grows with every turn and is re-sent in full on every LLM call.
messages: list = [{"role": "system", "content": SYSTEM_PROMPT}]

print(
    "Agent ready. Commands: /clear (forget conversation), /history (show message list), /quit"
)
print("-" * 60)

while True:
    try:
        user_input = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not user_input:
        continue

    if user_input == "/quit":
        print("Bye.")
        break

    if user_input == "/clear":
        # Resetting the list is all it takes to make the agent forget everything.
        # This is exactly what "clear conversation" does in any chat UI.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("Conversation cleared. Agent memory reset.")
        continue

    if user_input == "/history":
        # Print the raw message list — this is the agent's entire "memory".
        print("\n--- message history ---")
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content") or str(msg.get("tool_calls", ""))
            preview = content[:120].replace("\n", " ")
            print(f"  [{i}] {role}: {preview}")
        print(f"--- {len(messages)} messages total ---")
        continue

    messages.append({"role": "user", "content": user_input})
    answer = agent_step(messages)
    # agent_step already appended the assistant message to messages,
    # so we just print the answer — no need to append again.
    print(f"\nAgent: {answer}")
