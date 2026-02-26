"""Thoughtful AI Customer Support Agent

FAISS nearest-neighbor search over OpenAI embeddings for knowledge base
retrieval, with async OpenAI LLM fallback for unmatched questions.

OpenAI embeddings + FAISS: dense embeddings capture semantic similarity,
FAISS scales to millions of entries. IndexFlatIP is used here for exact
search; for >1M entries, swap to IndexIVFFlat or IndexHNSWFlat.
"""

import json
from pathlib import Path

import click
import faiss
import gradio as gr
import numpy as np
from openai import AsyncOpenAI, OpenAI, OpenAIError

APP_DIR = Path(__file__).parent


def load_knowledge_base(kb_path: Path) -> list[dict[str, str]]:
    """Load entries from the knowledge base JSON file."""
    return json.loads(kb_path.read_text())["questions"]


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


def build_faiss_index(
    questions: list[str],
    client: OpenAI,
    embedding_model: str,
) -> faiss.IndexFlatIP:
    """Embed questions and build a FAISS inner-product index.

    Vectors are L2-normalized, so inner product == cosine similarity.
    """
    embeddings = embed_sync(client, embedding_model, questions)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def render_system_prompt(entries: list[dict[str, str]]) -> str:
    """Inject knowledge base Q&A pairs into the prompt template."""
    template = APP_DIR.joinpath("system_prompt.txt").read_text()
    qa_text = "\n\n".join(
        f"Q: {entry['question']}\nA: {entry['answer']}"
        for entry in entries
    )
    return template.replace("{{KNOWLEDGE_BASE}}", qa_text)


class Agent:
    """Chat agent backed by async FAISS retrieval and async OpenAI fallback."""

    def __init__(
        self,
        client: AsyncOpenAI,
        llm_model: str,
        embedding_model: str,
        index: faiss.IndexFlatIP,
        answers: list[str],
        threshold: float,
        system_prompt: str,
    ):
        self.client = client
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.index = index
        self.answers = answers
        self.threshold = threshold
        self.system_prompt = system_prompt

    async def search(self, query: str) -> str | None:
        """Embed query via async OpenAI, then FAISS nearest-neighbor lookup."""
        query_vec = await embed_async(self.client, self.embedding_model, [query])
        scores, indices = self.index.search(query_vec, 1)
        if scores[0][0] >= self.threshold:
            return self.answers[indices[0][0]]
        return None

    async def respond(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> str:
        """Route to async KB search or async LLM fallback."""
        if not message or not message.strip():
            return "Please enter a question and I'll do my best to help."

        kb_answer = await self.search(message)
        if kb_answer is not None:
            return kb_answer

        try:
            messages: list[dict[str, str]] = [
                {"role": "system", "content": self.system_prompt},
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
    help="Path to knowledge_base.json.",
)
@click.option(
    "--threshold", type=float, required=True,
    help="Cosine similarity threshold for KB matching.",
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
    entries = load_knowledge_base(kb_path)
    questions = [entry["question"] for entry in entries]
    answers = [entry["answer"] for entry in entries]

    # Sync client for one-time startup index build (no event loop yet)
    sync_client = OpenAI(api_key=api_key)
    index = build_faiss_index(questions, sync_client, embedding_model)

    # Async client for runtime search + chat (used inside Gradio's event loop)
    async_client = AsyncOpenAI(api_key=api_key)
    system_prompt = render_system_prompt(entries)

    agent = Agent(
        client=async_client,
        llm_model=model,
        embedding_model=embedding_model,
        index=index,
        answers=answers,
        threshold=threshold,
        system_prompt=system_prompt,
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
