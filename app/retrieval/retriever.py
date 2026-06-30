"""Retriever — interface + STUB cho Buổi 1.

QUAN TRỌNG (điểm sư phạm): Ở Buổi 1 retriever là NO-OP — luôn trả về danh sách rỗng.
Nghĩa là chatbot trả lời chỉ dựa vào kiến thức sẵn có của model (parametric knowledge),
KHÔNG dựa trên tài liệu thật → dễ "hallucinate" nội dung luật.

Đây là "lỗ hổng RAG" có chủ đích. Buổi 4-5 sẽ thay stub này bằng:
  - embeddings.py   (Buổi 4)
  - vectorstore.py  (Buổi 4 — Qdrant)
  - chunking.py + loader.py (Buổi 5)
và retrieve() sẽ trả về các chunk thật từ vector store.

Interface giữ NGUYÊN qua các buổi → code gọi (pipeline.py, prompts) không phải sửa.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievedChunk:
    """Một đoạn văn bản lấy được từ retrieval, kèm metadata.

    score và source sẽ có ý nghĩa thật từ Buổi 5; ở Buổi 1 không bao giờ được tạo ra.
    """

    text: str
    source: str = ""
    score: float = 0.0


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """STUB Buổi 1: chưa có vector store nên trả rỗng.

    Args:
        query: câu hỏi của người dùng.
        top_k: số chunk muốn lấy (sẽ dùng từ Buổi 5).

    Returns:
        [] ở Buổi 1. Từ Buổi 5: list các RetrievedChunk liên quan nhất.
    """
    _ = (query, top_k)  # chưa dùng — giữ chữ ký ổn định cho các buổi sau
    return []
