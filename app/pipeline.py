"""Pipeline orchestrator — nối retrieval (stub) + prompt + LLM.

Đây là "xương sống" RAG. Ở Buổi 1, bước retrieve trả [] nên thực chất chỉ là
chatbot thuần LLM. Từ Buổi 5, chỉ cần retriever.retrieve() trả chunk thật là
toàn bộ pipeline thành RAG — KHÔNG phải sửa file này.

Luồng (sẽ giàu dần qua các buổi):
    [guardrails.check_input]  ← Buổi 7
    → retrieve(query)          ← stub Buổi 1, thật từ Buổi 5
    → build_messages(q, ctx)
    → llm.chat / chat_stream
    [guardrails.check_output] ← Buổi 7
"""

from __future__ import annotations

from collections.abc import Iterator

from app.llm import completion
from app.llm.params import GenerationParams
from app.prompts.templates import build_messages
from app.retrieval.retriever import retrieve
from app.schemas.domain import LegalAnswer


def answer(question: str, params: GenerationParams | None = None) -> str:
    """Trả lời dạng text (non-streaming)."""
    chunks = retrieve(question)  # Buổi 1: [] → trả lời thuần parametric knowledge
    messages = build_messages(question, chunks)
    return completion.chat(messages, params)


def answer_stream(
    question: str, params: GenerationParams | None = None
) -> Iterator[str]:
    """Trả lời dạng streaming (Section 6)."""
    chunks = retrieve(question)
    messages = build_messages(question, chunks)
    yield from completion.chat_stream(messages, params)


def answer_structured(
    question: str, params: GenerationParams | None = None
) -> LegalAnswer:
    """Trả lời dạng structured output theo schema LegalAnswer (Section 4)."""
    chunks = retrieve(question)
    messages = build_messages(question, chunks)
    return completion.chat_parsed(messages, LegalAnswer, params)
