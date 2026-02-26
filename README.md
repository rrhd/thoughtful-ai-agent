# Thoughtful AI Support Agent

Customer support chatbot for Thoughtful AI. Retrieves answers from a predefined knowledge base using TF-IDF similarity search, with OpenAI LLM fallback for general questions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI API key for LLM fallback (knowledge base lookups work without it):

```bash
export OPENAI_API_KEY="your-key-here"
```

## Run

```bash
python app.py
```

Opens at http://localhost:7860

## How it works

1. User sends a question via the Gradio chat UI
2. TF-IDF cosine similarity scores the question against 5 predefined Q&A pairs
3. Score above 0.3: returns the predefined answer directly
4. Score below 0.3: forwards to OpenAI `gpt-4o-mini` with the knowledge base as system prompt context
5. If OpenAI is unavailable: returns a friendly fallback message

### Why TF-IDF

The knowledge base has 5 entries. TF-IDF provides effective keyword-overlap matching at this scale without requiring embedding model downloads or additional API calls for the retrieval step. For a larger knowledge base, sentence embeddings or a vector database would be more appropriate.
