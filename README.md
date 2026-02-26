# Thoughtful AI Support Agent

Gradio chat UI backed by FAISS semantic retrieval (OpenAI embeddings) with async OpenAI LLM fallback.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python app.py \
  --model gpt-4o-mini \
  --embedding-model text-embedding-3-small \
  --api-key YOUR_KEY \
  --port 7860 \
  --kb-path knowledge_base.json \
  --threshold 0.4
```

## Architecture

OpenAI embeddings encode the knowledge base questions into a FAISS `IndexFlatIP` index at startup (sync). At runtime, incoming queries are embedded async and matched via cosine similarity (inner product on L2-normalized vectors). Above `--threshold`: return the predefined answer. Below: async OpenAI chat call with the full knowledge base injected as system prompt context. For >1M entries, swap `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSWFlat` for approximate search.
