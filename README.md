# Thoughtful AI Support Agent

Gradio chat UI backed by TF-IDF knowledge base retrieval with async OpenAI LLM fallback.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python app.py --model gpt-4o-mini --api-key YOUR_KEY --port 7860
```

## Architecture

TF-IDF cosine similarity against 5 predefined Q&A pairs. Above threshold: predefined answer. Below: async OpenAI call with the knowledge base injected as system prompt context. TF-IDF chosen over embeddings because the corpus is 5 items and stop word filtering already separates relevant (0.4+) from irrelevant (0.0) queries cleanly.
