"""Domain schemas — Bài 1, Section 4 (Structured Outputs với Pydantic).

Định nghĩa hình dạng câu trả lời của trợ lý pháp lý. Dùng cho structured-output mode
(client.chat.completions.parse). Field validators ép constraint ở model level.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Một trích dẫn nguồn. Ở Buổi 1 thường rỗng (chưa có retrieval);
    từ Buổi 5 sẽ map tới chunk thực tế trong vector store."""

    source: str = Field(description="Tên/định danh nguồn tài liệu")
    quote: str = Field(description="Đoạn trích liên quan từ tài liệu")


class LegalAnswer(BaseModel):
    """Câu trả lời có cấu trúc của trợ lý pháp lý."""

    answer: str = Field(description="Câu trả lời chính, ngắn gọn bằng tiếng Việt")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Độ tin cậy 0–1 của câu trả lời"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Các nguồn trích dẫn (rỗng nếu không có tài liệu)"
    )
    needs_lawyer: bool = Field(
        default=False, description="True nếu vụ việc nên tham khảo luật sư"
    )
