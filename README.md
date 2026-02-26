# Thoughtful AI Support Agent

Customer support chat agent with FAISS semantic search and LLM fallback.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp config.example.yaml config.yaml
```

Edit `config.yaml` with your API base URL and key, then:

```bash
python app.py
```

Opens at http://localhost:7860.

## How it works

Knowledge base (`knowledge_base.jsonl`) is embedded at startup into a FAISS HNSW index. Queries are embedded async, FAISS returns top-k neighbors. Above threshold: direct answer. Below: top-k results injected as context into an LLM call (RAG).

## Design choices

Because this should scale, I used FAISS (even though we have only a few examples here) and also ensured it's easy to add new examples.
