"""State schema — Buổi 6 (Agentic RAG / CRAG), LangGraph.

State là "bảng dữ liệu chung" chảy qua các node. Mỗi node nhận state, trả về
một phần state cần cập nhật (partial dict) — LangGraph tự merge vào state chính.
"""

from __future__ import annotations

from typing import TypedDict

from app.retrieval.retriever import RetrievedChunk


class GradedChunk(TypedDict):
    """Một chunk kèm kết quả chấm điểm liên quan (grade_documents node)."""

    chunk: RetrievedChunk
    relevant: bool
    reason: str


class GraphState(TypedDict, total=False):
    """State xuyên suốt graph CRAG + Query Decomposition.

    total=False: node chỉ cần trả field nó thực sự cập nhật, không phải khai
    báo đủ mọi key mỗi lần.
    """

    question: str                    # câu hỏi gốc của người dùng
    sub_questions: list[str]         # (nếu decompose) các câu hỏi con
    documents: list[RetrievedChunk]  # chunk đã retrieve (đã gộp nếu multi-hop)
    graded: list[GradedChunk]        # kết quả grading từng chunk
    web_search_used: bool            # có fallback web search không (để hiển thị UI)
    generation: str                  # câu trả lời cuối cùng
