# Thoughtful AI Support Agent

Customer support chatbot for Thoughtful AI. Retrieves answers from a predefined knowledge base using TF-IDF similarity search, with OpenAI LLM fallback for general questions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py --model gpt-4o-mini --api-key YOUR_KEY --port 7860
```

All flags:

| Flag | Description |
|------|-------------|
| `--model` | OpenAI model name (required) |
| `--api-key` | OpenAI API key (required) |
| `--port` | Port to serve on (required) |
| `--share` | Create a public Gradio share link |

## How it works

1. User sends a question via the Gradio chat UI
2. TF-IDF cosine similarity scores the question against 5 predefined Q&A pairs
3. Score above 0.3: returns the predefined answer directly
4. Score below 0.3: forwards to OpenAI (async) with the knowledge base as system prompt context
5. If OpenAI is unavailable: returns a friendly fallback message

### Why TF-IDF

The knowledge base has 5 entries. TF-IDF with English stop word filtering provides effective keyword-overlap matching at this scale without requiring embedding model downloads or additional API calls for the retrieval step. Irrelevant queries score 0.0, relevant paraphrases score 0.4+.
