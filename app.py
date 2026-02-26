"""Thoughtful AI Customer Support Agent

FAISS nearest-neighbor search over sentence embeddings for knowledge base
retrieval, with async OpenAI LLM fallback for unmatched questions.

Sentence embeddings + FAISS chosen over TF-IDF: dense embeddings capture
semantic similarity (not just keyword overlap), and FAISS scales to millions
of entries via approximate nearest-neighbor search. IndexFlatIP is used here
for exact search; for >1M entries, swap to IndexIVFFlat or IndexHNSWFlat.
"""

import json
from pathlib import Path

import click
import faiss
import gradio as gr
import numpy as np
from openai import AsyncOpenAI, OpenAIError
from sentence_transformers import SentenceTransformer

APP_DIR = Path(__file__).parent


def load_knowledge_base(kb_path: Path) -> tuple[list[str], list[str]]:
    """Load questions and answers from the knowledge base JSON file."""
    data = json.loads(kb_path.read_text())
    questions = [entry["question"] for entry in data["questions"]]
    answers = [entry["answer"] for entry in data["questions"]]
    return questions, answers


def build_faiss_index(
    questions: list[str],
    encoder: SentenceTransformer,
) -> faiss.IndexFlatIP:
    """Encode questions and build a FAISS inner-product index.

    Vectors are L2-normalized, so inner product == cosine similarity.
    """
    embeddings = encoder.encode(
        questions, normalize_embeddings=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def render_system_prompt(kb_path: Path) -> str:
    """Load the prompt template and inject knowledge base Q&A pairs."""
    data = json.loads(kb_path.read_text())
    template = APP_DIR.joinpath("system_prompt.txt").read_text()
    qa_text = "\n\n".join(
        f"Q: {entry['question']}\nA: {entry['answer']}"
        for entry in data["questions"]
    )
    return template.replace("{{KNOWLEDGE_BASE}}", qa_text)


class Agent:
    """Chat agent backed by FAISS retrieval and async OpenAI fallback."""

    def __init__(
        self,
        client: AsyncOpenAI,
        llm_model: str,
        encoder: SentenceTransformer,
        index: faiss.IndexFlatIP,
        answers: list[str],
        threshold: float,
        system_prompt: str,
    ):
        self.client = client
        self.llm_model = llm_model
        self.encoder = encoder
        self.index = index
        self.answers = answers
        self.threshold = threshold
        self.system_prompt = system_prompt

    def search(self, query: str) -> str | None:
        """FAISS nearest-neighbor lookup against the knowledge base."""
        query_vec = self.encoder.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)
        scores, indices = self.index.search(query_vec, 1)
        if scores[0][0] >= self.threshold:
            return self.answers[indices[0][0]]
        return None

    async def respond(
        self,
        message: str,
        history: list[dict[str, str]],
    ) -> str:
        """Route to KB search or async LLM fallback."""
        if not message or not message.strip():
            return "Please enter a question and I'll do my best to help."

        kb_answer = self.search(message)
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
@click.option("--model", type=str, required=True, help="OpenAI model name.")
@click.option("--api-key", type=str, required=True, help="OpenAI API key.")
@click.option("--port", type=int, required=True, help="Port to serve on.")
@click.option(
    "--encoder-model", type=str, required=True,
    help="Sentence-transformers model name for KB retrieval.",
)
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
    api_key: str,
    port: int,
    encoder_model: str,
    kb_path: Path,
    threshold: float,
    share: bool,
) -> None:
    questions, answers = load_knowledge_base(kb_path)
    encoder = SentenceTransformer(encoder_model)
    index = build_faiss_index(questions, encoder)
    system_prompt = render_system_prompt(kb_path)
    client = AsyncOpenAI(api_key=api_key)

    agent = Agent(
        client=client,
        llm_model=model,
        encoder=encoder,
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
