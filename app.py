"""Thoughtful AI Customer Support Agent

FAISS nearest-neighbor search over OpenAI embeddings for knowledge base
retrieval, with async OpenAI LLM fallback (RAG pattern) for unmatched
questions.

OpenAI embeddings + FAISS: dense embeddings capture semantic similarity,
FAISS scales to millions of entries. IndexFlatIP is used here for exact
search; for >1M entries, swap to IndexIVFFlat or IndexHNSWFlat.
"""

import json
from pathlib import Path
from typing import Generator

import click
import faiss
import gradio as gr
import numpy as np
from openai import AsyncOpenAI, OpenAI, OpenAIError

APP_DIR = Path(__file__).parent

# OpenAI embeddings API accepts up to 2048 inputs per request
EMBEDDING_BATCH_SIZE = 2048

# Best match checked against threshold; remaining top-k used as RAG
# context for LLM fallback
RETRIEVAL_TOP_K = 3


def stream_knowledge_base(
    kb_path: Path,
) -> Generator[dict[str, str], None, None]:
    """Yield entries one at a time from a JSONL knowledge base file."""
    with kb_path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def embed_sync(
    client: OpenAI,
    model: str,
    texts: list[str],
) -> np.ndarray:
    """Embed texts via OpenAI (sync) and return L2-normalized float32 matrix."""
    response = client.embeddings.create(model=model, input=texts)
    matrix = np.array(
        [item.embedding for item in response.data], dtype=np.float32,
    )
    faiss.normalize_L2(matrix)
    return matrix


async def embed_async(
    client: AsyncOpenAI,
    model: str,
    texts: list[str],
) -> np.ndarray:
    """Embed texts via OpenAI (async) and return L2-normalized float32 matrix."""
    response = await client.embeddings.create(model=model, input=texts)
    matrix = np.array(
        [item.embedding for item in response.data], dtype=np.float32,
    )
    faiss.normalize_L2(matrix)
    return matrix


def flush_embedding_batch(
    client: OpenAI,
    embedding_model: str,
    batch: list[str],
    index: faiss.IndexFlatIP | None,
) -> faiss.IndexFlatIP:
    """Embed a batch and add to the FAISS index, creating it if needed."""
    embeddings = embed_sync(client, embedding_model, batch)
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def build_faiss_index(
    kb_path: Path,
    client: OpenAI,
    embedding_model: str,
) -> tuple[faiss.IndexFlatIP, list[str], list[str]]:
    """Stream JSONL, embed in batches, build FAISS index incrementally.

    Returns (index, questions, answers). Questions and answers are parallel
    lists aligned with the FAISS index row order.
    """
    questions: list[str] = []
    answers: list[str] = []
    index: faiss.IndexFlatIP | None = None
    batch: list[str] = []

    for entry in stream_knowledge_base(kb_path):
        questions.append(entry["question"])
        answers.append(entry["answer"])
        batch.append(entry["question"])
        if len(batch) >= EMBEDDING_BATCH_SIZE:
            index = flush_embedding_batch(client, embedding_model, batch, index)
            batch.clear()

    if batch:
        index = flush_embedding_batch(client, embedding_model, batch, index)

    if index is None:
        raise ValueError(f"No entries found in {kb_path}")

    return index, questions, answers


class Agent:
    """Chat agent backed by async FAISS retrieval and async OpenAI RAG fallback."""

    def __init__(
        self,
        client: AsyncOpenAI,
        llm_model: str,
        embedding_model: str,
        index: faiss.IndexFlatIP,
        questions: list[str],
        answers: list[str],
        threshold: float,
        prompt_template: str,
    ):
        self.client = client
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.index = index
        self.questions = questions
        self.answers = answers
        self.threshold = threshold
        self.prompt_template = prompt_template

    async def search(
        self,
        query: str,
    ) -> list[tuple[str, str, float]]:
        """Return top-k (question, answer, score) tuples from FAISS."""
        query_vec = await embed_async(self.client, self.embedding_model, [query])
        scores, indices = self.index.search(query_vec, RETRIEVAL_TOP_K)
        results: list[tuple[str, str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((
                    self.questions[idx],
                    self.answers[idx],
                    float(score),
                ))
        return results

    async def respond(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> str:
        """Route to async KB search or async LLM RAG fallback."""
        if not message or not message.strip():
            return "Please enter a question and I'll do my best to help."

        results = await self.search(message)

        if results and results[0][2] >= self.threshold:
            return results[0][1]

        # RAG: inject top-k retrieval results as LLM context
        context = "\n\n".join(
            f"Q: {question}\nA: {answer}"
            for question, answer, score in results
        )
        prompt = self.prompt_template.replace("{{CONTEXT}}", context)

        try:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": prompt},
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": message})
            response = await self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
            )
            return response.choices[0].message.content
        except OpenAIError:
            return (
                "I couldn't find a matching answer in our knowledge base, "
                "and my AI assistant is currently unavailable. "
                "Please try again later or contact support@thoughtful.ai."
            )
        except Exception:
            return "Something went wrong. Please try again."


@click.command()
@click.option("--model", type=str, required=True, help="OpenAI chat model name.")
@click.option(
    "--embedding-model", type=str, required=True,
    help="OpenAI embedding model name.",
)
@click.option("--api-key", type=str, required=True, help="OpenAI API key.")
@click.option("--port", type=int, required=True, help="Port to serve on.")
@click.option(
    "--kb-path", type=click.Path(exists=True, path_type=Path), required=True,
    help="Path to knowledge base JSONL file.",
)
@click.option(
    "--threshold", type=float, required=True,
    help="Cosine similarity threshold for direct KB answers.",
)
@click.option("--share", is_flag=True, help="Create a public Gradio share link.")
def main(
    model: str,
    embedding_model: str,
    api_key: str,
    port: int,
    kb_path: Path,
    threshold: float,
    share: bool,
) -> None:
    # Sync client for one-time startup index build (no event loop yet)
    sync_client = OpenAI(api_key=api_key)
    index, questions, answers = build_faiss_index(kb_path, sync_client, embedding_model)

    # Async client for runtime search + chat (used inside Gradio's event loop)
    async_client = AsyncOpenAI(api_key=api_key)
    prompt_template = APP_DIR.joinpath("system_prompt.txt").read_text()

    agent = Agent(
        client=async_client,
        llm_model=model,
        embedding_model=embedding_model,
        index=index,
        questions=questions,
        answers=answers,
        threshold=threshold,
        prompt_template=prompt_template,
    )

    demo = gr.ChatInterface(
        fn=agent.respond,
        title="Thoughtful AI Support Agent",
        description=(
            "Ask me about Thoughtful AI's healthcare automation agents, "
            "including EVA, CAM, and PHIL."
        ),
        examples=[
            "What does EVA do?",
            "Tell me about your agents",
            "What are the benefits of using Thoughtful AI?",
        ],
    )
    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    main()
