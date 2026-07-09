"""Embeddings — Buổi 4 (Embeddings & Vector Databases).

Dùng OpenAI text-embedding-3-small (1536 dims). Gọi API qua native SDK.

Khác với model e5 (cần prefix query/passage), OpenAI embedding KHÔNG cần prefix —
dùng thẳng text. Query và passage embed giống nhau, nên chỉ cần một hàm lõi.

Lưu ý: embedding luôn dùng OpenAI Cloud (cần OPENAI_API_KEYS), độc lập với
LLM_BACKEND. Kể cả khi chat chạy trên Ollama local, embedding vẫn gọi OpenAI —
đó là lý do dùng client riêng ở đây thay vì llm/client.get_client().
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_client():
    """OpenAI client cho embedding (luôn Cloud). Cache lại."""
    from openai import OpenAI

    keys = settings.api_keys
    if not keys:
        raise ValueError(
            "Embedding cần OPENAI_API_KEYS (OpenAI Cloud). Đặt trong .env."
        )
    return OpenAI(api_key=keys[0])


def _embed(texts: list[str]) -> list[list[float]]:
    """Gọi OpenAI embeddings API cho một batch text."""
    response = _get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed tài liệu/đoạn văn. Dùng khi index."""
    return _embed(texts)


def embed_query(text: str) -> list[float]:
    """Embed câu truy vấn. Dùng khi search."""
    return _embed([text])[0]


def embedding_dim() -> int:
    """Số chiều vector (text-embedding-3-small = 1536)."""
    return settings.embedding_dim
