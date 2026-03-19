"""
02_embeddings.py — Generate and explore text embeddings.

Demonstrates:
- Loading a local sentence-transformers model (no API key needed)
- Embedding sentences into fixed-size float vectors
- Computing cosine similarity to measure semantic closeness
- Observing that similar meaning → high similarity, regardless of exact wording
"""

from sentence_transformers import SentenceTransformer

# ── Model ─────────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 80 MB, 384-dim vectors, runs fast on CPU.
# First run downloads it to ~/.cache/huggingface/; subsequent runs are instant.

model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Sentences ─────────────────────────────────────────────────────────────────
# Three groups: two semantically similar pairs + one outlier.

sentences = [
    "Python is a popular programming language.",  # 0 — about Python
    "I love coding in Python.",  # 1 — about Python (different wording)
    "Git helps teams collaborate on code.",  # 2 — about Git
    "Version control tracks changes over time.",  # 3 — about Git (different wording)
    "The weather today is sunny and warm.",  # 4 — unrelated
]

# ── Embed ─────────────────────────────────────────────────────────────────────

embeddings = model.encode(sentences, normalize_embeddings=True)
# Shape: (num_sentences, embedding_dim)
print(f"Embedding shape: {embeddings.shape}")
print(f"Each sentence → {embeddings.shape[1]}-dimensional float vector\n")

# ── Cosine similarity ─────────────────────────────────────────────────────────
# Since embeddings are L2-normalised, dot product == cosine similarity

similarity = embeddings @ embeddings.T  # shape: (5, 5)

print("=== Cosine similarity matrix ===")
print("     ", "  ".join(f"[{i}]" for i in range(len(sentences))))
for i, row in enumerate(similarity):
    scores = "  ".join(f"{v:.2f}" for v in row)
    print(f"  [{i}]  {scores}")

print("\nSentences:")
for i, s in enumerate(sentences):
    print(f"  [{i}] {s}")

print("\n=== Key observations ===")
pairs = [
    (0, 1, "Python pair"),
    (2, 3, "Git pair"),
    (0, 4, "Python vs weather"),
    (2, 4, "Git vs weather"),
]
for i, j, label in pairs:
    print(f"  {label}: {similarity[i, j]:.3f}")
