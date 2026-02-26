"""Thoughtful AI Customer Support Agent"""
import gradio as gr
from openai import OpenAI, OpenAIError
from sklearn.feature_extraction.text import TfidfVectorizer

KNOWLEDGE_BASE = [
    {
        "question": "What does the eligibility verification agent (EVA) do?",
        "answer": (
            "EVA automates the process of verifying a patient's eligibility "
            "and benefits information in real-time, eliminating manual data "
            "entry errors and reducing claim rejections."
        ),
    },
    {
        "question": "What does the claims processing agent (CAM) do?",
        "answer": (
            "CAM streamlines the submission and management of claims, "
            "improving accuracy, reducing manual intervention, and "
            "accelerating reimbursements."
        ),
    },
    {
        "question": "How does the payment posting agent (PHIL) work?",
        "answer": (
            "PHIL automates the posting of payments to patient accounts, "
            "ensuring fast, accurate reconciliation of payments and "
            "reducing administrative burden."
        ),
    },
    {
        "question": "Tell me about Thoughtful AI's Agents.",
        "answer": (
            "Thoughtful AI provides a suite of AI-powered automation agents "
            "designed to streamline healthcare processes. These include "
            "Eligibility Verification (EVA), Claims Processing (CAM), and "
            "Payment Posting (PHIL), among others."
        ),
    },
    {
        "question": "What are the benefits of using Thoughtful AI's agents?",
        "answer": (
            "Using Thoughtful AI's Agents can significantly reduce "
            "administrative costs, improve operational efficiency, and "
            "reduce errors in critical processes like claims management "
            "and payment posting."
        ),
    },
]

QUESTIONS = [entry["question"] for entry in KNOWLEDGE_BASE]
ANSWERS = [entry["answer"] for entry in KNOWLEDGE_BASE]

vectorizer = TfidfVectorizer(stop_words="english")
question_vectors = vectorizer.fit_transform(QUESTIONS)

# With English stop words filtered, irrelevant queries score 0.0 and relevant
# paraphrases score 0.4+. Any threshold in that gap works; 0.3 is conservative.
SIMILARITY_THRESHOLD = 0.3


def search_knowledge_base(query: str) -> str | None:
    """Find the best predefined answer via TF-IDF cosine similarity.

    Returns the answer text if similarity >= threshold, else None.
    """
    query_vector = vectorizer.transform([query])
    scores = (query_vector @ question_vectors.T).toarray().ravel()
    best_idx = scores.argmax()
    if scores[best_idx] >= SIMILARITY_THRESHOLD:
        return ANSWERS[best_idx]
    return None


def build_system_prompt() -> str:
    """Embed the full knowledge base into the LLM system prompt."""
    qa_text = "\n\n".join(
        f"Q: {entry['question']}\nA: {entry['answer']}"
        for entry in KNOWLEDGE_BASE
    )
    return (
        "You are a customer support agent for Thoughtful AI, a company that "
        "builds AI-powered automation agents for healthcare revenue cycle "
        "management.\n\n"
        "Official product information:\n\n"
        f"{qa_text}\n\n"
        "Use this information when answering questions about Thoughtful AI. "
        "For unrelated questions, respond helpfully using general knowledge. "
        "Be professional and concise."
    )


SYSTEM_PROMPT = build_system_prompt()


def get_llm_response(
    message: str,
    history: list[dict[str, str]],
) -> str:
    """Call OpenAI for questions not matched by the knowledge base."""
    client = OpenAI()
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return response.choices[0].message.content


def respond(
    message: str,
    history: list[dict[str, str]],
) -> str:
    """Route user input to knowledge base search or LLM fallback."""
    if not message or not message.strip():
        return "Please enter a question and I'll do my best to help."

    kb_answer = search_knowledge_base(message)
    if kb_answer is not None:
        return kb_answer

    try:
        return get_llm_response(message, history)
    except OpenAIError:
        return (
            "I couldn't find a matching answer in our knowledge base, "
            "and my AI assistant is currently unavailable. "
            "Please try again later or contact support@thoughtful.ai."
        )
    except Exception:
        return "Something went wrong. Please try again."


demo = gr.ChatInterface(
    fn=respond,
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

if __name__ == "__main__":
    demo.launch()
