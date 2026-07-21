"""State schema — Buổi 6 (Agentic RAG / CRAG), LangGraph.

State là "bảng dữ liệu chung" chảy qua các node. Mỗi node nhận state, trả về
một phần state cần cập nhật (partial dict) — LangGraph tự merge vào state chính.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.retrieval.retriever import RetrievedChunk


class GradedChunk(TypedDict):
    """Một chunk kèm kết quả chấm điểm liên quan (grade_documents node)."""

    chunk: RetrievedChunk
    relevant: bool
    reason: str


class RetrieveTask(TypedDict):
    """Payload gửi qua Send() tới node parallel_retrieve — mỗi task 1 sub-question."""

    sub_question: str


class GraphState(TypedDict, total=False):
    """State xuyên suốt graph CRAG + Query Decomposition.

    total=False: node chỉ cần trả field nó thực sự cập nhật, không phải khai
    báo đủ mọi key mỗi lần.

    QUAN TRỌNG cho Send API (fan-out song song) — HAI field tách biệt:

      - `raw_documents`: reducer `operator.add` (Annotated). Mỗi nhánh Send
        (1 sub-question/nhánh) chạy song song và trả {"raw_documents": [...]};
        LangGraph NỐI các list lại (a + b + c) khi hội tụ, không ghi đè.

      - `documents`: KHÔNG có reducer (ghi đè bình thường). Đây là field
        `dedupe_documents` ghi ra sau khi gộp raw_documents — và cũng là field
        mà grade/web_search/generate đọc/ghi tiếp theo.

    LÝ DO PHẢI TÁCH: nếu `documents` vừa là đích nhận từ Send vừa có reducer
    `operator.add`, thì mỗi khi `dedupe_documents` trả kết quả đã gộp, reducer
    lại CỘNG THÊM (không ghi đè) vào giá trị cũ — gây nhân bản dữ liệu mỗi lần
    node chạy lại trong một fan-in. Tách field mới tránh được bẫy này.
    """

    question: str                    # câu hỏi gốc của người dùng
    sub_questions: list[str]         # (nếu decompose) các câu hỏi con
    sub_question: str                # dùng nội bộ bởi mỗi nhánh Send (1 sub-question)
    raw_documents: Annotated[list[RetrievedChunk], operator.add]  # fan-in từ Send
    documents: list[RetrievedChunk]  # kết quả sau dedupe; đọc/ghi bởi các node sau
    graded: list[GradedChunk]        # kết quả grading từng chunk
    web_search_used: bool            # có fallback web search không (để hiển thị UI)
    generation: str                  # câu trả lời cuối cùng
