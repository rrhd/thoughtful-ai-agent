# Thoughtful AI Support Agent

Gradio chat UI backed by FAISS semantic retrieval (OpenAI embeddings) with async OpenAI RAG fallback.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python app.py \
  --model gpt-4o-mini \
  --embedding-model text-embedding-3-small \
  --api-key YOUR_KEY \
  --port 7860 \
  --kb-path knowledge_base.jsonl \
  --threshold 0.4
```

## Architecture

Knowledge base is JSONL (one entry per line), streamed at startup. Questions are embedded in batches via OpenAI and added to a FAISS `IndexFlatIP` index incrementally. At runtime, queries are embedded async, and FAISS returns the top-k nearest neighbors. Above `--threshold`: return the predefined answer directly. Below: inject top-k results as context into the LLM prompt (RAG pattern). For >1M entries, swap `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSWFlat`.
