"""
01_hello_llm.py — First API call to Ollama or OpenRouter.

Demonstrates:
- Configuring the backend via BACKEND env var ("openrouter" or "ollama")
- A basic non-streaming chat completion request
- Reading the raw response structure
- A streaming request, printing tokens as they arrive
"""

import json
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
else:  # ollama — uses its OpenAI-compatible endpoint
    BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/v1"
    MODEL = os.getenv("OLLAMA_MODEL")
    HEADERS = {"Content-Type": "application/json"}

CHAT_URL = f"{BASE_URL}/chat/completions"

# ── Non-streaming call ────────────────────────────────────────────────────────

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 2 + 2? Answer in one sentence."}],
}

response = requests.post(CHAT_URL, headers=HEADERS, json=payload)
response.raise_for_status()
data = response.json()

print("=== Raw response structure ===")
print(json.dumps(data, indent=2))
print("\n=== Extracted answer ===")
print(data["choices"][0]["message"]["content"])

# ── Streaming call ────────────────────────────────────────────────────────────

payload["stream"] = True

print("\n=== Streaming tokens ===")
with requests.post(CHAT_URL, headers=HEADERS, json=payload, stream=True) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8")
        if not text.startswith("data: "):
            continue  # skip SSE comments and other non-data lines
        text = text[6:]
        if not text or text == "[DONE]":
            break
        chunk = json.loads(text)
        token = chunk["choices"][0]["delta"].get("content", "")
        print(token, end="", flush=True)
print()
