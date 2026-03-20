"""
01_lcel_basics.py — LCEL chain syntax and composition.

Demonstrates:
- Creating a chat model (ChatOllama or ChatOpenAI for OpenRouter)
- Building a PromptTemplate → Model → OutputParser chain with the | operator
- Chain composition: piping one chain's output into another
- Streaming tokens from a chain
"""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ── Model setup ──────────────────────────────────────────────────────────────

BACKEND = os.getenv("BACKEND", "openrouter")

if BACKEND == "openrouter":
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=os.getenv("OPENROUTER_MODEL"),
    )
else:
    from langchain_ollama import ChatOllama

    model = ChatOllama(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("OLLAMA_MODEL"),
    )

parser = StrOutputParser()

# ── 1. Simple chain: prompt | model | parser ────────────────────────────────
# LCEL uses the | operator to pipe components together, just like a Unix pipe.
# Each component transforms the data and passes it to the next.

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in exactly two sentences, for a beginner."
)

chain = prompt | model | parser

print("=== Simple chain ===")
result = chain.invoke({"topic": "what a REST API is"})
print(result)

# ── 2. Composition: chain one output into another ───────────────────────────
# You can build a second chain that takes the first chain's output as input.

translate_prompt = ChatPromptTemplate.from_template(
    "Translate the following text to Portuguese (Brazil):\n\n{text}"
)

# RunnablePassthrough isn't needed — just pipe the output string into a dict.
composed = (
    chain  # returns a string (the explanation)
    | (lambda text: {"text": text})  # wrap it for the next prompt's variable
    | translate_prompt
    | model
    | parser
)

print("\n=== Composed chain (explain → translate) ===")
result = composed.invoke({"topic": "what a REST API is"})
print(result)

# ── 3. Streaming ────────────────────────────────────────────────────────────
# .stream() yields tokens as they arrive instead of waiting for the full response.

print("\n=== Streaming ===")
for token in chain.stream({"topic": "what embeddings are in machine learning"}):
    print(token, end="", flush=True)
print()
