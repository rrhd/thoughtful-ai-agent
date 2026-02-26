# Thoughtful AI Support Agent

Gradio chat UI backed by FAISS semantic retrieval with async OpenAI LLM fallback.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python app.py \
  --model gpt-4o-mini \
  --api-key YOUR_KEY \
  --port 7860 \
  --encoder-model all-MiniLM-L6-v2 \
  --kb-path knowledge_base.json \
  --threshold 0.4
```

## Architecture

Sentence embeddings (via `--encoder-model`) encode the knowledge base questions into a FAISS `IndexFlatIP` index at startup. Incoming queries are encoded and matched via cosine similarity (inner product on L2-normalized vectors). Above `--threshold`: return the predefined answer. Below: async OpenAI call with the full knowledge base injected as system prompt context. For >1M entries, swap `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSWFlat` for approximate search.
