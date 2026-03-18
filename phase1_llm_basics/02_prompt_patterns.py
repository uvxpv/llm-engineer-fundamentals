"""
02_prompt_patterns.py — Prompt engineering patterns.

Demonstrates:
- System prompts: shaping model persona and behavior
- Few-shot prompting: in-context learning from examples
- Chain-of-thought: reasoning before answering
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

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


def chat(messages: list[dict]) -> str:
    response = requests.post(
        CHAT_URL, headers=HEADERS, json={"model": MODEL, "messages": messages}
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── 1. System prompt ──────────────────────────────────────────────────────────
# The system role sets a persistent persona or constraint for the whole conversation.

answer = chat([
    {
        "role": "system",
        "content": (
            "You are a terse Linux expert. Respond only with the exact command needed, "
            "no explanation, no markdown."
        ),
    },
    {"role": "user", "content": "How do I list all files including hidden ones?"},
])

print("=== System prompt ===")
print(answer)

# ── 2. Few-shot prompting ─────────────────────────────────────────────────────
# Provide example input→output pairs in the conversation so the model learns
# the expected format from context, without any fine-tuning.

answer = chat([
    {"role": "user", "content": "Sentiment: 'I love this product!'"},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "Sentiment: 'This is the worst experience ever.'"},
    {"role": "assistant", "content": "negative"},
    {"role": "user", "content": "Sentiment: 'It arrived on time.'"},
])

print("\n=== Few-shot prompting ===")
print(answer)

# ── 3. Chain-of-thought ───────────────────────────────────────────────────────
# Instructing the model to reason step by step before giving a final answer
# improves accuracy on problems that require multi-step reasoning.

answer = chat([
    {
        "role": "system",
        "content": "Think through the problem step by step, then give a final answer.",
    },
    {
        "role": "user",
        "content": (
            "A bat and a ball cost $1.10 in total. "
            "The bat costs $1.00 more than the ball. "
            "How much does the ball cost?"
        ),
    },
])

print("\n=== Chain-of-thought ===")
print(answer)
