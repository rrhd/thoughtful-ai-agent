"""Thoughtful AI customer support agent."""

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import yaml
from openai import AsyncOpenAI, OpenAIError

APP_DIR = Path(__file__).parent


def read_jsonl(path: Path) -> Iterator[dict[str, str]]:
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


class Agent:

    @classmethod
    async def create(cls, config: dict) -> "Agent":
        agent = cls(config)
        agent.questions, agent.answers, agent.index = await agent.build_index()
        return agent

    def __init__(self, config: dict):
        self.client = AsyncOpenAI(
            base_url=config["openai"]["base_url"],
            api_key=config["openai"]["api_key"],
        )
        self.embedding_model = config["openai"]["embedding_model"]
        self.chat_model = config["openai"]["chat_model"]
        self.embedding_batch_size = config["retrieval"]["embedding_batch_size"]
        self.hnsw_m = config["retrieval"]["hnsw_m"]
        self.threshold = config["retrieval"]["threshold"]
        self.top_k = config["retrieval"]["top_k"]
        self.kb_path = Path(config["retrieval"]["kb_path"])

        pc = yaml.safe_load(
            APP_DIR.joinpath(config["retrieval"]["prompt_template_path"]).read_text(),
        )["system"]
        instructions = "\n".join(f"- {item}" for item in pc["instructions"])
        self.system_prefix = (
            pc["prompt"]
            .replace("{{identity}}", pc["identity"].strip())
            .replace("{{instructions}}", instructions)
        )
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.index: faiss.IndexHNSWFlat | None = None

    @staticmethod
    def to_matrix(embedding_data: list) -> np.ndarray:
        matrix = np.array(
            [item.embedding for item in embedding_data], dtype=np.float32,
        )
        faiss.normalize_L2(matrix)
        return matrix

    async def build_index(self):
        questions, answers = zip(
            *((e["question"], e["answer"]) for e in read_jsonl(self.kb_path)),
        )
        questions, answers = list(questions), list(answers)

        batches = [questions[i:i + self.embedding_batch_size] for i in range(0, len(questions), self.embedding_batch_size)]
        responses = await asyncio.gather(
            *(self.client.embeddings.create(model=self.embedding_model, input=batch) for batch in batches),
        )

        all_embeddings = np.vstack([self.to_matrix(r.data) for r in responses])
        index = faiss.IndexHNSWFlat(all_embeddings.shape[1], self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.add(all_embeddings)

        return questions, answers, index

    async def embed(self, texts: list[str]) -> np.ndarray:
        response = await self.client.embeddings.create(
            model=self.embedding_model, input=texts,
        )
        return self.to_matrix(response.data)

    async def search(self, query: str) -> list[tuple[str, str, float]]:
        query_vec = await self.embed([query])
        scores, indices = self.index.search(query_vec, self.top_k)
        return [
            (self.questions[i], self.answers[i], float(s))
            for s, i in zip(scores[0], indices[0])
        ]

    async def respond(self, message: str, history: list[dict[str, str]]) -> str:
        results = await self.search(message)

        if results and results[0][2] >= self.threshold:
            return results[0][1]

        # FAISS returns descending by score, no re-sort needed
        context = "\n\n".join(
            f"Q: {question}\nA: {answer}"
            for question, answer, _score in results
        )
        prompt = self.system_prefix.replace("{{context}}", context)

        try:
            messages = [{"role": "system", "content": prompt}]
            messages.extend(history)
            messages.append({"role": "user", "content": message})
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
            )
            return response.choices[0].message.content
        except OpenAIError:
            return "Our AI assistant is currently unavailable. Please try again later."


async def main() -> None:
    config = yaml.safe_load(APP_DIR.joinpath("config.yaml").read_text())
    agent = await Agent.create(config)
    demo = gr.ChatInterface(
        fn=agent.respond,
        title=config["server"]["title"],
        description=config["server"]["description"],
        examples=config["server"]["examples"],
    )
    demo.launch(server_port=config["server"]["port"], share=config["server"]["share"])


if __name__ == "__main__":
    asyncio.run(main())
