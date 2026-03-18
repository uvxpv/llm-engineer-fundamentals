"""
03_structured_output.py — JSON mode + Pydantic validation.

Demonstrates:
- Defining a Pydantic schema and embedding it in the prompt
- Using response_format={"type":"json_object"} to enforce JSON output
- Parsing and validating the response with Pydantic
"""

import json
import os

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from typing import Literal

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


# ── Schema ────────────────────────────────────────────────────────────────────

class MovieReview(BaseModel):
    title: str
    sentiment: Literal["positive", "negative", "neutral"]
    score: int          # 1–10
    summary: str        # one sentence


EXAMPLE = json.dumps({
    "title": "Movie Title",
    "sentiment": "positive | negative | neutral",
    "score": 8,
    "summary": "One sentence summary.",
}, indent=2)

REVIEW_TEXT = (
    "Dune Part Two was visually stunning and the performances were excellent. "
    "The pacing dragged a little in the middle but the finale was worth it."
)

# ── Request ───────────────────────────────────────────────────────────────────

payload = {
    "model": MODEL,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a movie critic assistant. "
                "Respond ONLY with a JSON object in exactly this format:\n\n"
                f"{EXAMPLE}\n\n"
                "score must be an integer from 1 to 10. "
                "sentiment must be exactly one of: positive, negative, neutral."
            ),
        },
        {"role": "user", "content": f"Review:\n{REVIEW_TEXT}"},
    ],
    "response_format": {"type": "json_object"},
}

response = requests.post(CHAT_URL, headers=HEADERS, json=payload)
response.raise_for_status()
raw_json = response.json()["choices"][0]["message"]["content"]

print("=== Raw JSON from model ===")
print(raw_json)

# ── Validation ────────────────────────────────────────────────────────────────

try:
    review = MovieReview.model_validate_json(raw_json)
    print("\n=== Validated Pydantic object ===")
    print(review)
    print(f"\nTitle: {review.title}")
    print(f"Sentiment: {review.sentiment}")
    print(f"Score: {review.score}/10")
    print(f"Summary: {review.summary}")
except ValidationError as e:
    print("\n=== Validation failed ===")
    print(e)
