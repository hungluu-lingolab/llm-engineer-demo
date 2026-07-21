"""Nodes cho Agentic RAG (CRAG + Query Decomposition) — Buổi 6.

Mỗi node là một hàm thuần: nhận GraphState, trả về phần state cần cập nhật.
QUAN TRỌNG: mọi lời gọi bên trong đều dùng NATIVE SDK đã xây từ các buổi trước —
  - completion.chat / chat_parsed  (Buổi 1)
  - retriever.retrieve              (Buổi 5)
  - tools_web.web_search            (Buổi 6, Tavily native)
LangGraph chỉ lo control flow (node nào chạy sau node nào), không gọi LangChain.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.agent.state import GradedChunk, GraphState
from app.agent.tools_web import web_search
from app.config import settings
from app.llm import completion
from app.llm.params import GenerationParams
from app.retrieval.retriever import RetrievedChunk, retrieve


# ── Query Decomposition ───────────────────────────────────────────────────────

class _SubQuestions(BaseModel):
    sub_questions: list[str] = Field(
        description="2-4 câu hỏi con, mỗi câu trả lời độc lập được"
    )


def decompose_query(state: GraphState) -> dict:
    """Chia câu hỏi phức tạp thành sub-questions (structured output).

    Chỉ decompose khi câu hỏi đủ dài (heuristic AGENT_DECOMPOSE_MIN_CHARS) —
    tránh tốn 1 lời gọi LLM cho câu hỏi đơn giản.
    """
    question = state["question"]
    if len(question) < settings.agent_decompose_min_chars:
        return {"sub_questions": [question]}

    messages = [
        {
            "role": "system",
            "content": (
                "Chia câu hỏi pháp lý phức tạp sau thành 2-4 sub-questions đơn giản "
                "hơn, mỗi câu trả lời độc lập được bằng tra cứu luật. Nếu câu hỏi đã "
                "đơn giản, trả về chính nó trong danh sách 1 phần tử."
            ),
        },
        {"role": "user", "content": question},
    ]
    result = completion.chat_parsed(
        messages, _SubQuestions, GenerationParams(temperature=0.0)
    )
    sub_qs = result.sub_questions[: settings.agent_max_sub_questions] or [question]
    return {"sub_questions": sub_qs}


# ── Retrieve (multi-hop nếu có nhiều sub-question) ───────────────────────────

def retrieve_node(state: GraphState) -> dict:
    """Retrieve cho từng sub-question rồi gộp + khử trùng (giữ thứ tự, ưu tiên score cao)."""
    sub_questions = state.get("sub_questions") or [state["question"]]

    seen: dict[str, RetrievedChunk] = {}
    for sub_q in sub_questions:
        for chunk in retrieve(sub_q):
            existing = seen.get(chunk.text)
            if existing is None or chunk.score > existing.score:
                seen[chunk.text] = chunk

    documents = sorted(seen.values(), key=lambda c: c.score, reverse=True)
    return {"documents": documents}


# ── Grade documents ───────────────────────────────────────────────────────────

class _Grade(BaseModel):
    relevant: bool = Field(description="Tài liệu có liên quan trực tiếp tới câu hỏi không")
    reason: str = Field(description="Giải thích ngắn gọn vì sao relevant/irrelevant")


def grade_documents(state: GraphState) -> dict:
    """Chấm điểm liên quan cho từng chunk bằng structured output (chính xác hơn parse free-text)."""
    question = state["question"]
    graded: list[GradedChunk] = []

    for chunk in state.get("documents", []):
        messages = [
            {
                "role": "system",
                "content": (
                    "Đánh giá xem TÀI LIỆU có liên quan trực tiếp để trả lời CÂU HỎI không."
                ),
            },
            {
                "role": "user",
                "content": f"Câu hỏi: {question}\n\nTài liệu: {chunk.text}",
            },
        ]
        grade = completion.chat_parsed(messages, _Grade, GenerationParams(temperature=0.0))
        graded.append({"chunk": chunk, "relevant": grade.relevant, "reason": grade.reason})

    relevant_docs = [g["chunk"] for g in graded if g["relevant"]]
    return {"documents": relevant_docs, "graded": graded}


# ── Router: có cần web search fallback không? ────────────────────────────────

def route_after_grading(state: GraphState) -> str:
    """Conditional edge: đủ chunk relevant → generate thẳng; thiếu → web_search."""
    n_relevant = len(state.get("documents", []))
    if n_relevant >= settings.agent_min_relevant_chunks:
        return "generate"
    return "web_search"


# ── Web search fallback ───────────────────────────────────────────────────────

def web_search_node(state: GraphState) -> dict:
    """CRAG fallback: không đủ chunk relevant từ vector DB → tìm trên web."""
    results = web_search(state["question"])
    merged = state.get("documents", []) + results
    return {"documents": merged, "web_search_used": True}


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(state: GraphState) -> dict:
    """Sinh câu trả lời cuối cùng, bám vào documents đã có (grounded)."""
    from app.prompts.templates import build_messages

    messages = build_messages(state["question"], state.get("documents", []))
    answer = completion.chat(messages, GenerationParams(temperature=0.1))
    return {"generation": answer}
