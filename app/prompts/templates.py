"""Prompt templates — Bài 1, Section 3 (Prompt Patterns).

Minh hoạ role prompting (persona luật sư VN) + chỗ chèn context cho RAG (Buổi 5).
Ở Buổi 1, context luôn rỗng (retriever là stub) — system prompt được thiết kế để
xử lý cả trường hợp "không có tài liệu" một cách trung thực.
"""

from __future__ import annotations

from app.retrieval.retriever import RetrievedChunk

# Role prompting (Section 3): persona cụ thể, có quy tắc rõ ràng.
LEGAL_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về luật doanh nghiệp Việt Nam.

Quy tắc:
- Trả lời chính xác, ngắn gọn, bằng tiếng Việt.
- Khi có TÀI LIỆU THAM KHẢO bên dưới, CHỈ trả lời dựa trên tài liệu đó và trích dẫn nguồn.
- Khi KHÔNG có tài liệu tham khảo, nói rõ rằng câu trả lời dựa trên hiểu biết chung
  và khuyến nghị người dùng kiểm chứng với văn bản luật chính thức.
- Luôn khuyên tham khảo luật sư cho các vụ việc cụ thể.
"""


def build_messages(
    question: str, context_chunks: list[RetrievedChunk] | None = None
) -> list[dict]:
    """Dựng danh sách messages cho một câu hỏi.

    Nếu có context_chunks (từ Buổi 5 trở đi) thì chèn vào trước câu hỏi —
    đây chính là bước "Augment" của RAG. Ở Buổi 1, context_chunks rỗng.
    """
    system_content = LEGAL_SYSTEM_PROMPT

    if context_chunks:
        context_block = "\n\n".join(
            f"[Nguồn {i + 1}] {c.text}" for i, c in enumerate(context_chunks)
        )
        system_content += f"\n\n--- TÀI LIỆU THAM KHẢO ---\n{context_block}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question},
    ]
