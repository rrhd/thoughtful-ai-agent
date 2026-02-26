"""Thoughtful AI customer support agent."""

import json
from collections.abc import Iterator
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import yaml
from openai import AsyncOpenAI, OpenAI, OpenAIError

APP_DIR = Path(__file__).parent


def read_jsonl(path: Path) -> Iterator[dict[str, str]]:
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


class Agent:
    def __init__(self, config: dict):
        self.config = config
        self.async_client = AsyncOpenAI(api_key=config["openai"]["api_key"])
        self.prompt_template = APP_DIR.joinpath(
            config["retrieval"]["prompt_template_path"],
        ).read_text()
        self.questions, self.answers, self.index = self.build_index()

    @staticmethod
    def to_matrix(embedding_data: list) -> np.ndarray:
        matrix = np.array(
            [item.embedding for item in embedding_data], dtype=np.float32,
        )
        faiss.normalize_L2(matrix)
        return matrix

    def build_index(self):
        client = OpenAI(api_key=self.config["openai"]["api_key"])
        model = self.config["openai"]["embedding_model"]
        batch_size = self.config["retrieval"]["embedding_batch_size"]

        questions = []
        answers = []
        for entry in read_jsonl(Path(self.config["retrieval"]["kb_path"])):
            questions.append(entry["question"])
            answers.append(entry["answer"])

        index = None
        for i in range(0, len(questions), batch_size):
            embeddings = self.to_matrix(
                client.embeddings.create(
                    model=model, input=questions[i:i + batch_size],
                ).data,
            )
            if index is None:
                index = faiss.IndexHNSWFlat(
                    embeddings.shape[1],
                    self.config["retrieval"]["hnsw_m"],
                    faiss.METRIC_INNER_PRODUCT,
                )
            index.add(embeddings)

        return questions, answers, index

    async def embed(self, texts: list[str]) -> np.ndarray:
        response = await self.async_client.embeddings.create(
            model=self.config["openai"]["embedding_model"], input=texts,
        )
        return self.to_matrix(response.data)

    async def search(self, query: str) -> list[tuple[str, str, float]]:
        query_vec = await self.embed([query])
        scores, indices = self.index.search(
            query_vec, self.config["retrieval"]["top_k"],
        )
        return [
            (self.questions[idx], self.answers[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
        ]

    async def respond(self, message: str, history: list[dict[str, str]]) -> str:
        results = await self.search(message)

        if results and results[0][2] >= self.config["retrieval"]["threshold"]:
            return results[0][1]

        context = "\n\n".join(
            f"Q: {question}\nA: {answer}"
            for question, answer, score in results
        )
        prompt = self.prompt_template.replace("{{CONTEXT}}", context)

        try:
            messages = [{"role": "system", "content": prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})
            response = await self.async_client.chat.completions.create(
                model=self.config["openai"]["chat_model"],
                messages=messages,
            )
            return response.choices[0].message.content
        except OpenAIError:
            return "Our AI assistant is currently unavailable. Please try again later."


def main() -> None:
    config = yaml.safe_load(APP_DIR.joinpath("config.yaml").read_text())
    agent = Agent(config)
    demo = gr.ChatInterface(
        fn=agent.respond,
        title=config["server"]["title"],
        description=config["server"]["description"],
        examples=config["server"]["examples"],
    )
    demo.launch(server_port=config["server"]["port"], share=config["server"]["share"])


if __name__ == "__main__":
    main()
