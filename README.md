# Thoughtful AI Support Agent

FAISS semantic retrieval (OpenAI embeddings) with async OpenAI RAG fallback. Gradio chat UI.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Edit `config.yaml` with your OpenAI API key, then:

```bash
python app.py
```

## Architecture

Knowledge base is JSONL, streamed at startup. Questions are embedded in batches and added to a FAISS `IndexFlatIP` incrementally. At runtime, queries are embedded async, FAISS returns top-k neighbors. Above threshold: direct answer. Below: top-k injected as context into the LLM prompt (RAG). For >1M entries, swap `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSWFlat`.
