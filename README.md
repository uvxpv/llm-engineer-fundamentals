# LLM Engineer Fundamentals

A structured, hands-on learning project for modern AI engineering. Each phase builds on the last, taking you from raw LLM API calls to evaluated RAG pipelines and autonomous agents.

## Learning Path

| Phase | Topic | What You'll Build |
|-------|-------|-------------------|
| 1 | LLM Basics | Raw HTTP calls to Ollama, structured output with Pydantic, prompt patterns |
| 2 | RAG from Scratch | Chunking, embeddings, vector store, and a full retrieval pipeline — no frameworks |
| 3 | LangChain | Rebuild the RAG pipeline using LangChain and LCEL chains |
| 4 | Agents | ReAct agent pattern with tool use |
| 5 | Evaluation | Measure RAG quality with RAGAS metrics |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed locally

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd llm-engineer-fundamentals

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys if using OpenRouter

# Pull the Ollama model
ollama pull llama3.1:8b
```

## Running Scripts

Each phase directory contains numbered Python scripts. Run them in order:

```bash
python phase1_llm_basics/01_hello_llm.py
```
