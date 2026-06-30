"""API request/response schemas (FastAPI layer).

Tách riêng khỏi schemas/domain.py: đây là "hợp đồng" HTTP với client,
còn domain.py là hình dạng dữ liệu nội bộ. Giữ tách biệt để hai thứ tiến hoá độc lập.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.config import settings


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="Câu hỏi của người dùng")
    # Cho phép override generation params mỗi request (Bài 1, Section 2).
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_completion_tokens: int | None = Field(default=None, gt=0)


class ChatResponse(BaseModel):
    answer: str
    model: str = Field(default_factory=lambda: settings.llm_model)
